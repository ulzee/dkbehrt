# %%
import argparse
import os, sys
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='base')
parser.add_argument('--use_embedding', type=str, default=None)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--nowandb', action='store_true', default=False)
parser.add_argument('--mask_ratio', type=float, default=0.15)
parser.add_argument('--disable_visit_shuffle', action='store_true', default=False)
args = parser.parse_args()
#%%
if not args.nowandb:
    os.environ["WANDB_PROJECT"] = "icd"
    os.environ["WANDB_LOG_MODEL"] = "end"
else:
    os.environ['WANDB_DISABLED'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import torch
import torch.nn as nn
import pickle as pk
import numpy as np
from transformers import BertConfig, BertForMaskedLM
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import embeddings
import utils
#%%
with open('saved/diagnoses.pk', 'rb') as fl:
    dxs = pk.load(fl)
#%%
tokenizer = AutoTokenizer.from_pretrained('./saved/tokenizers/bert')
#%%
bert_emb_size = 192
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
if args.mode == 'emb':
    assert args.use_embedding is not None
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
    els -= np.mean(els, axis=0)
    els /= np.std(els, axis=0)

    print(f'Loaded embeddings for {nmatched}/{len(tokenizer.vocab)}')

    if els.shape[1] <= bertconfig.hidden_size:
        els = np.concatenate([els, np.zeros((len(els), bertconfig.hidden_size - els.shape[1]))], axis=1)
    else:
        raise 'Not handled'
    model.bert.embeddings = embeddings.InjectEmbeddings(bertconfig, els, keep_training=False)

#%%
if args.mode == 'emb':
    param_list = [t[1] for t in model.named_parameters() if 'word_embeddings' not in t[0]]
else:
    param_list = list(model.parameters())
optimizer = torch.optim.AdamW(
    param_list,
    lr=args.lr,
)
# %%
phase_ids = { phase: np.genfromtxt(f'artifacts/splits/{phase}_ids.txt') for phase in ['train', 'val', 'test'] }
phase_ids['val'] = phase_ids['val'][::10][:1024]
datasets = { phase: utils.ICDDataset(
    dxs,
    tokenizer,
    ids,
    separator='[SEP]',
    max_length=512,
    shuffle_in_visit=False if args.disable_visit_shuffle else phase=='train',
) for phase, ids in phase_ids.items() }

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mask_ratio,
)

mdlname = f'bert-{args.mode}-layers{args.layers}-h{args.heads}'
training_args = TrainingArguments(
    output_dir=f'runs/{mdlname}',
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    report_to='wandb' if not args.nowandb else None,
    evaluation_strategy='steps',
    run_name=mdlname,
    eval_steps=500,
    save_steps=1000,
)

def compute_metrics(eval_pred, mask_value=-100, topns=(1, 5, 10)):
    logits, labels = eval_pred
    bsize, seqlen = labels.shape

    logits = torch.from_numpy(np.reshape(logits, (bsize*seqlen, -1)))
    labels = torch.from_numpy(np.reshape(labels, (bsize*seqlen)))
    where_prediction = labels != mask_value

    topaccs = utils.topk_accuracy(logits[where_prediction], labels[where_prediction], topk=topns)

    return { f'top{n:02d}': acc for n, acc in zip(topns, topaccs) }

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['val'],
    compute_metrics=compute_metrics,
)
#%%
trainer.evaluate()
trainer.train()
# %%
torch.save(model.state_dict(), 'saved/bert_basic.pth')
# %%
