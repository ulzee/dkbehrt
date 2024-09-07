# %%
import argparse
import os, sys
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--load', type=str, required=True)
parser.add_argument('--scratch', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--code_resolution', type=int, default=5)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--eval_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=2000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--disable_visible_devices', action='store_true', default=False)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--ehr_outcomes_path', type=str, default='../ehr-outcomes')
parser.add_argument('--nowandb', action='store_true', default=False)
parser.add_argument('--predict', action='store_true', default=False)
parser.add_argument('--predict_set', default='test', type=str)
parser.add_argument('--train_subsample', default=None, type=int)
parser.add_argument('--val_subsample', default=10, type=int)
parser.add_argument('--eval_test', default=False, action='store_true')
parser.add_argument('--covariates', default='gender,age', type=str)
parser.add_argument('--silent', default=False, action='store_true')
parser.add_argument('--use_embedding', type=str, default=None)
parser.add_argument('--emb_proj', type=str, default='linear')
args = parser.parse_args()
#%%
if not args.nowandb:
    os.environ["WANDB_PROJECT"] = f'ehr-outcomes-{args.dataset}-{args.task}'
    os.environ["WANDB_LOG_MODEL"] = "end"
else:
    os.environ['WANDB_DISABLED'] = 'true'
if not args.disable_visible_devices:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import torch
import torch.nn as nn
import pickle as pk
import numpy as np
from transformers.utils import logging
from transformers import TrainerCallback
from transformers import AutoTokenizer, TrainingArguments, Trainer, BertForSequenceClassification
import embedding
import attention
import utils
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import random, string
import collator
if args.silent: logging.set_verbosity_error()
#%%
run_tag = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
model_mode = args.load.split('/')[-2].split('-')[1] # expects model_name/checkpoint_num folders
#%%
with open(f'saved/{args.dataset}/diagnoses-cr{args.code_resolution}.pk', 'rb') as fl:
    dxs = pk.load(fl)
#%%
covs = utils.load_covariates(f'saved/{args.dataset}/cov.csv', covlist=args.covariates.split(','))
#%%
tokenizer = AutoTokenizer.from_pretrained(f'./saved/{args.dataset}/tokenizers/bert-cr{args.code_resolution}')
tokenizer._pad = lambda *args, **kwargs: collator._pad(tokenizer, *args, **kwargs)
#%%
loaded_name = args.load.split('/')[-2]
scratchtag = ('pretrained_frozen-' if args.freeze else 'pretrained-') if not args.scratch else f'scratch-'
subtag = '' if args.subsample is None else f'sub{args.subsample}-'
mdlname = f'ft-{subtag}{scratchtag}{args.task}-ftlr{args.lr}-{loaded_name}'
if not args.silent: print(mdlname)

model = BertForSequenceClassification.from_pretrained(args.load, num_labels=2)

def load_embedding_file(fname):
    with open(fname, 'rb') as fl:
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

    print(f'Loaded embeddings for {nmatched}/{len(tokenizer.vocab)}')

    return els

if model_mode == 'base':
    model = BertForSequenceClassification(model.config)
    model.bert.embeddings = embedding.CovariateAddEmbeddings(config=model.config)
    if not args.scratch:
        model.load_state_dict(torch.load(f'{args.load}/weights.pth'), strict=False)
elif model_mode == 'attn':

    els = None
    if args.use_embedding == 'zeros':
        els = np.zeros((len(tokenizer.vocab), 100))
        print(f'Loading all-zero embeddings (for debugging)')
    elif '.txt' not in args.use_embedding:
        els = load_embedding_file(args.use_embedding)
        emb_dict_list = [els]
    elif '.txt' in args.use_embedding:
        emb_dict_list = []
        with open(args.use_embedding) as fl:
            for efilename in fl:
                efilename = efilename.strip()
                emb_dict_list += [load_embedding_file(efilename)]

    # reset model for now
    model = BertForSequenceClassification.from_pretrained(args.load, num_labels=2)
    bertconfig = model.config
    if model_mode == 'emb':
        if els.shape[1] <= bertconfig.hidden_size:
            els = np.concatenate([els, np.zeros((len(els), bertconfig.hidden_size - els.shape[1]))], axis=1)
        else:
            raise 'Not handled'

        model.bert.embeddings = embedding.InjectEmbeddings(bertconfig, els, keep_training=False)

    elif model_mode == 'attn':
        # NOTE: some variables are wrapped in a non-module class to prevent issue
        #  with safetensors trying to save duplicate (shared) variables

        model.bert.embeddings = embedding.KeepInputEmbeddings(config=bertconfig)

        emb_holders = []
        for ei, ith_embset in enumerate(emb_dict_list):
            extra_embeddings = nn.Embedding(*ith_embset.shape)
            extra_embeddings.weight = nn.Parameter(torch.from_numpy(ith_embset.astype(np.float32)).cuda(), requires_grad=False)
            emb_holders += [
                embedding.NonTorchVariableHolder(
                    extra_embeddings=extra_embeddings)
            ]

        for layer in model.bert.encoder.layer:
            layer.attention.self = attention.WeightedAttention(
                config=bertconfig,
                embeddings=emb_holders, # NOTE: now expects a list (should be list of one item for one emb set)
                current_input=model.bert.embeddings.input_ids,
                use_proj=args.emb_proj)

        if not args.silent: print('Attached weighted attention layers.')

    if not args.scratch:
        model.load_state_dict(torch.load(f'{args.load}/weights.pth'), strict=False)
        # NOTE: until we can figure out a way to make hug trainer save custom models, we need to load from a state dict
#%%
phase_ids = { phase: np.genfromtxt(f'files/{args.dataset}/{phase}_ids.txt').astype(int) for phase in ['train', 'val', 'test'] }
if args.subsample is not None:
    phase_ids['train'] = phase_ids['train'][::args.subsample]
if not args.predict:
    # reduce val set to not slow down training
    phase_ids['val'] = phase_ids['val'][::args.val_subsample]
if args.train_subsample is not None:
    phase_ids['train'] = phase_ids['train'][:args.train_subsample]

datasets = { phase: utils.EHROutcomesDataset(
    args.task,
    f'{args.ehr_outcomes_path}/saved/{args.dataset}',
    tokenizer,
    ids,
    covs,
    max_length=512,
    code_resolution=args.code_resolution, # WARN: the benchmark may not parse to the same resolution
    shuffle_in_visit=phase=='train',
    verbose=not args.silent,
) for phase, ids in phase_ids.items() }

training_args = TrainingArguments(
    output_dir=f'runs/{mdlname}',
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_accumulation_steps=15,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    report_to='wandb' if not args.nowandb else None,
    eval_strategy='steps',
    run_name=mdlname,
    eval_steps=args.eval_steps,
    save_steps=args.save_steps,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pos_labels = labels[:, 1]
    predictions = np.argmax(logits, axis=-1)

    accuracy = (predictions == pos_labels).sum() / len(predictions)

    probs = torch.sigmoid(torch.from_numpy(logits[:, 1])).numpy()
    roc = roc_auc_score(pos_labels, probs)
    ap = average_precision_score(pos_labels, probs)
    f1 = f1_score(pos_labels, probs > 0.5, average='macro')

    unique_preds = len(np.unique(logits[:, 1]))

    return dict(
        accuracy=accuracy,
        pr=ap,
        roc=roc,
        f1=f1,
        unique_preds=unique_preds,
    )

class CustomCallback(TrainerCallback):

    best_loss = None

    def on_log(self, __args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            # any custom logging
            if 'eval_val_loss' in logs:
                if self.best_loss is None or self.best_loss > logs['eval_val_loss']:
                    self.best_loss = logs['eval_val_loss']

                    print('saved ckpt:', logs['eval_val_loss'])
                    torch.save(model.state_dict(), f'saved/{args.dataset}/best_{mdlname}.pth')

if args.freeze:
    train_params = ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
    for name, param in model.named_parameters():
        if name not in train_params:
            param.requires_grad = False

#%%
if not args.predict:
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset={ ph: datasets[ph] for ph in ['val'] + [[], ['test']][args.eval_test]},
        compute_metrics=compute_metrics,
        callbacks=[CustomCallback()]
    )

    trainer.evaluate()
    trainer.train()
else:
    if args.task in ['mortality', 'los72']:
        with open(f'../ehr-outcomes/files/{args.dataset}/boot_ixs.pk', 'rb') as fl:
            boot_ixs = pk.load(fl)
    else:
        with open(f'../ehr-outcomes/files/{args.dataset}/boot_ixs_{args.task}.pk', 'rb') as fl:
            boot_ixs = pk.load(fl)

    def get_boot_metrics(ypred):
        ls = []
        ytrue = np.array(datasets[args.predict_set].labels)
        for bxs in [list(range(len(ytrue)))] + boot_ixs[:10]:
            ls += [[
                average_precision_score(ytrue[bxs], ypred[bxs]),
                roc_auc_score(ytrue[bxs], ypred[bxs]),
                f1_score(ytrue[bxs], ypred[bxs] > 0.5, average='micro'),
            ]]

        out = ''
        for tag, replicates in zip(['ap', 'roc', 'f1'], zip(*ls)):
            est, replicates = replicates[0]*100, replicates[1:]
            ci = 1.95*np.std(np.array(replicates)*100)
            out += f'{tag}:{est:.2f} ({ci:.2f}) '

        print(out)


    model.load_state_dict(torch.load(f'saved/{args.dataset}/best_{mdlname}.pth'))
    tester = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
    )
    preds = tester.predict(datasets[args.predict_set])

    class_proba = torch.sigmoid(torch.from_numpy(preds.predictions[:, 1]))

    get_boot_metrics(class_proba)

    # for met in [roc_auc_score, average_precision_score, f1_score]:
    #     if met == f1_score:
    #         print(str(met), met(datasets[args.predict_set].labels, class_proba > 0.5))
    #     else:
    #         print(str(met), met(datasets[args.predict_set].labels, class_proba))
# %%
