# %%
import argparse
import os, sys
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--use_embedding', type=str, default=None)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embdim', type=int, default=192)
parser.add_argument('--train_subsample', default=None, type=int)
parser.add_argument('--val_subsample', type=int, default=10)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--eval_steps', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--max_steps', type=int, default=None)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--nowandb', action='store_true', default=False)
parser.add_argument('--mask_ratio', type=float, default=0.5)
parser.add_argument('--code_resolution', type=int, default=5)
parser.add_argument('--disable_visit_shuffle', action='store_true', default=False)
parser.add_argument('--disable_visible_devices', action='store_true', default=False)
parser.add_argument('--covariates', default='gender,age', type=str)
args = parser.parse_args()
#%%
if not args.nowandb:
    wandb_subtag = '' if args.train_subsample is None else '-sub'
    os.environ["WANDB_PROJECT"] = f'icdbert-{args.dataset}{wandb_subtag}'
    os.environ["WANDB_LOG_MODEL"] = "end"
    import wandb
else:
    os.environ['WANDB_DISABLED'] = 'true'
if not args.disable_visible_devices:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
# else:
#     torch.cuda_set_device(int(os.environ['CUDA_VISIBLE_DEVICES']))
import torch
import torch.nn as nn
import pickle as pk
import numpy as np
from transformers import BertConfig, BertForMaskedLM, TrainerCallback
from transformers.integrations import WandbCallback
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import collator
from collator import CustomDataCollatorForLanguageModeling
from torch.utils.data import Dataset
import embedding
import pandas as pd
import attention
import utils
import random, string
run_tag = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
#%%
with open(f'saved/{args.dataset}/diagnoses-cr{args.code_resolution}.pk', 'rb') as fl:
    dxs = pk.load(fl)
#%%
covs = utils.load_covariates(f'saved/{args.dataset}/cov.csv', covlist=args.covariates.split(','))
#%%
tokenizer = AutoTokenizer.from_pretrained(f'./saved/{args.dataset}/tokenizers/bert-cr{args.code_resolution}')
tokenizer._pad = lambda *args, **kwargs: collator._pad(tokenizer, *args, **kwargs)
#%%
bert_emb_size = args.embdim
bertconfig = BertConfig(
    vocab_size=len(tokenizer.vocab),
    max_position_embeddings=tokenizer.model_max_length,
    hidden_size=bert_emb_size,
    num_hidden_layers=args.layers,
    num_attention_heads=args.heads,
    intermediate_size=1024,
)
model = BertForMaskedLM(bertconfig)
#%%
if args.mode == 'base':
    model.bert.embeddings = embedding.CovariateAddEmbeddings(config=bertconfig)
elif args.mode in ['emb', 'attn']:
    assert args.use_embedding is not None

    els = None
    if args.use_embedding == 'zeros':
        els = np.zeros((len(tokenizer.vocab), 100))
        print(f'Loading all-zero embeddings (for debugging)')
    else:
        with open(args.use_embedding, 'rb') as fl:
            edict = pk.load(fl)
        edim = len(next(iter(edict.values())))

        template = np.zeros((len(tokenizer.vocab), edim))
        nmatched = 0
        for w, i in tokenizer.vocab.items():
            w = w.upper()
            if w in edict:
                template[i] = edict[w]
                nmatched += 1
        els = np.array(template).astype(np.float32)
        els = np.array([v/np.sqrt(np.sum(v**2)) if np.sum(v != 0) != 0 else v for v in els])
        # els /= np.std(els, axis=0)

        print(f'Loaded embeddings for {nmatched}/{len(tokenizer.vocab)}')

    if args.mode == 'emb':
        if els.shape[1] <= bertconfig.hidden_size:
            els = np.concatenate([els, np.zeros((len(els), bertconfig.hidden_size - els.shape[1]))], axis=1)
        else:
            raise 'Not handled'

        model.bert.embeddings = embedding.InjectEmbeddings(bertconfig, els, keep_training=False)

    elif args.mode == 'attn':
        # NOTE: some variables are wrapped in a non-module class to prevent issue
        #  with safetensors trying to save duplicate (shared) variables

        model.bert.embeddings = embedding.KeepInputEmbeddings(config=bertconfig)

        extra_embeddings = nn.Embedding(*els.shape)
        extra_embeddings.weight = nn.Parameter(torch.from_numpy(els.astype(np.float32)).cuda(), requires_grad=False)
        embedding_dict_holder = embedding.NonTorchVariableHolder(
            extra_embeddings=extra_embeddings)

        for layer in model.bert.encoder.layer:
            layer.attention.self = attention.WeightedAttention(
                config=bertconfig,
                embeddings=embedding_dict_holder,
                current_input=model.bert.embeddings.input_ids)

        print('Attached weighted attention layers.')

# %%
phase_ids = { phase: np.genfromtxt(f'files/{args.dataset}/{phase}_ids.txt') for phase in ['train', 'val', 'test'] }
phase_ids['val'] = phase_ids['val'][::args.val_subsample]
if args.train_subsample is not None:
    phase_ids['train'] = phase_ids['train'][:args.train_subsample]
datasets = { phase: utils.ICDDataset(
    dxs,
    tokenizer,
    ids,
    covs,
    separator='[SEP]',
    max_length=512,
    shuffle_in_visit=False if args.disable_visit_shuffle else phase=='train',
) for phase, ids in phase_ids.items() }

token_frequency = dict()
for sample in datasets['val']:
    for tkn in sample['input_ids']:
        if tkn < 3: continue
        c = token_frequency.get(tkn, 0)
        token_frequency[tkn] = c + 1
least_frequent = dict()
tix, counts = list(zip(*token_frequency.items()))
prev_cval = 10
for cutoff in [50, 100, 200]:
    cval = cutoff
    least_frequent[f'least{cutoff}'] = [i for i, c in token_frequency.items() if c <= cval and c > prev_cval]
    print(f'Tokens <{cutoff}:', len(least_frequent[f'least{cutoff}']))
    prev_cval = cval

data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_ratio,
)

subtag = '' if args.train_subsample is None else f'sub{args.train_subsample}-'
mdlname = f'bert-{args.mode}-{subtag}cr{args.code_resolution}-lr{args.lr}-e{args.embdim}-layers{args.layers}-h{args.heads}_{run_tag}'
print(mdlname)
training_args = TrainingArguments(
    output_dir=f'runs/{args.dataset}/{mdlname}',
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_accumulation_steps=15,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    max_steps=-1 if args.max_steps is None else args.max_steps,
    report_to='wandb' if not args.nowandb else None,
    evaluation_strategy='steps',
    run_name=mdlname,
    eval_steps=args.eval_steps,
    save_steps=args.eval_steps,
)

def compute_metrics(eval_pred, mask_value=-100, topns=(1, 5, 10)):
    logits, labels = eval_pred
    bsize, seqlen = labels.shape

    logits = torch.from_numpy(np.reshape(logits, (bsize*seqlen, -1)))
    labels = torch.from_numpy(np.reshape(labels, (bsize*seqlen)))
    where_prediction = labels != mask_value

    topaccs = utils.topk_accuracy(logits[where_prediction], labels[where_prediction], topk=topns)
    out = dict()
    for n, acc in zip(topns, topaccs):
        out[f'top{n:02d}'] = acc

    if args.mode == 'emb':
        inspect_idx = tokenizer.vocab['f32'] if 'f32' in tokenizer.vocab else tokenizer.vocab['f329']
        cl = model.bert.embeddings.coef_learn[inspect_idx].item()
        out['coef_learn'] = cl

    logits = logits.cpu().numpy()
    labels = labels.cpu().numpy()
    for freqbin, tixs in least_frequent.items():
        idict = { i: True for i in tixs }
        where_bin = [i for i, l in enumerate(labels.astype(int).tolist()) if l in idict]
        out[freqbin+'_count'] = len(where_bin)
        out[freqbin] = utils.topk_accuracy(
            torch.from_numpy(logits[where_bin]),
            torch.from_numpy(labels[where_bin]),
            topk=[5])[0]

    return out

class CustomCallback(TrainerCallback):
    def on_log(self, __args, state, control, logs=None, **kwargs):
        # super().on_log(**kwargs)
        if state.is_local_process_zero:
            if args.mode == 'emb':
                coefs = model.bert.embeddings.coef_learn.detach().cpu().numpy()
                coefs = [[i] for i in coefs]
                table = wandb.Table(data=coefs, columns=["coefs"])
                wandb.log({
                    'histogram-coef_learn': wandb.plot.histogram(table, "coefs", title="Embedding mixing coefficient")
                })

            if state.global_step > 0:
                ckpt_path = f'runs/{args.dataset}/{mdlname}/checkpoint-{state.global_step}'
                if not os.path.exists(ckpt_path):
                    os.mkdir(ckpt_path)
                torch.save(
                    model.state_dict(),
                    f'{ckpt_path}/weights.pth')

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['val'],
    compute_metrics=compute_metrics,
    callbacks=[CustomCallback()]
)
#%%
trainer.evaluate()
trainer.train()
# %%
