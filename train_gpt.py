# %%
import argparse
import os, sys
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='llama')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--nowandb', action='store_true', default=True)
args = parser.parse_args()
#%%
if not args.nowandb:
    os.environ["WANDB_PROJECT"] = "icd"
    os.environ["WANDB_LOG_MODEL"] = "end"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
import torch
import pickle as pk
import numpy as np
from transformers import MistralConfig, MistralForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import utils
#%%
config_class, model_class = dict(
    mistral=(MistralConfig, MistralForCausalLM),
    llama=(LlamaConfig, LlamaForCausalLM),
)[args.arch]
#%%
with open('saved/diagnoses.pk', 'rb') as fl:
    dxs = pk.load(fl)
#%%
tokenizer = AutoTokenizer.from_pretrained('./saved/tokenizers/gpt')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
#%%
mdlconfig = config_class(
    vocab_size=len(tokenizer.vocab),
    hidden_size=192,
    num_hidden_layers=args.layers,
    num_attention_heads=args.heads,
    intermediate_size=1024,
)
model = model_class(mdlconfig)
#%%
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
)
# %%
class ICDDataset(Dataset):
    def __init__(self, patient_ids):
        pdict = { i: True for i in patient_ids }
        self.pids = [i for i in dxs if i in pdict]
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, i):
        # FIXME: code cutoff id diagnoses file?
        concepts = dxs[self.pids[i]]['codes']['concepts']
        visits = dxs[self.pids[i]]['codes']['visits']
        vid = visits[0]

        code_series = ''
        for c, v in zip(concepts, visits):
            if v != vid:
                vid = v
                code_series += ' <v>'
            code_series += f' {c}'

        return { k: v for k, v in tokenizer(code_series, padding=True).items()  if k not in ['token_type_ids']}
phase_ids = { phase: np.genfromtxt(f'artifacts/splits/{phase}_ids.txt') for phase in ['train', 'val', 'test'] }
phase_ids['val'] = phase_ids['val'][::10][:1024]
datasets = { phase: ICDDataset(ids) for phase, ids in phase_ids.items() }

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

training_args = TrainingArguments(
    output_dir=f'runs/gpt-{args.arch}',
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=16,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    report_to='wandb' if not args.nowandb else None,
    evaluation_strategy='steps',
    run_name=f'gpt-{args.arch}',
    eval_steps=500,
    save_steps=1000,
)

def compute_metrics(eval_pred, mask_value=-100, topks=(1, 5, 10)):
    logits, labels = eval_pred
    bsize, seqlen = labels.shape

    logits = torch.from_numpy(np.reshape(logits, (bsize*seqlen, -1)))
    labels = torch.from_numpy(np.reshape(labels, (bsize*seqlen)))
    where_prediction = labels != mask_value

    topaccs = utils.topk_accuracy(logits[where_prediction], labels[where_prediction], topk=topks)

    return { f'top{n:02d}': acc for n, acc in zip(topks, topaccs) }

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=datasets['train'],
    eval_dataset=datasets['val'],
    compute_metrics=compute_metrics,
)
# %%
trainer.evaluate()
trainer.train()
# # %%
torch.save(model.state_dict(), 'saved/gpt_basic.pth')
# # %%
