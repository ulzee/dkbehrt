# %%
import argparse
import os, sys
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--load', type=str, required=True)
parser.add_argument('--scratch', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--code_resolution', type=int, default=5)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--eval_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--disable_visible_devices', action='store_true', default=False)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--ehr_outcomes_path', type=str, default='../ehr-outcomes')
parser.add_argument('--nowandb', action='store_true', default=False)
args = parser.parse_args()
#%%
if not args.nowandb:
    os.environ["WANDB_PROJECT"] = "ehr-outcomes"
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
from transformers import AutoTokenizer, TrainingArguments, Trainer, BertForSequenceClassification
from torch.utils.data import Dataset
import embedding
import attention
import utils
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import random, string
import evaluate
metric = evaluate.load("accuracy")
run_tag = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
#%%
with open(f'saved/diagnoses-cr{args.code_resolution}.pk', 'rb') as fl:
    dxs = pk.load(fl)
#%%
tokenizer = AutoTokenizer.from_pretrained(f'./saved/tokenizers/bert-cr{args.code_resolution}')
#%%
model = BertForSequenceClassification.from_pretrained(args.load, num_labels=2)

if args.scratch:
    model = BertForSequenceClassification(model.config)
#%%

# either base bert approach or a custom model
model_mode = args.load.split('/')[-2].split('-')[1] # expects model_name/checkpoint_num folders
#%%
# FIXME: this is not used at all?
if model_mode in ['emb', 'attn']:
    # These are embeddings passed by the user, they should not be backpropd
    param_list = [t[1] for t in model.named_parameters() if 'extra_embeddings' not in t[0]]
else:
    param_list = list(model.parameters())
optimizer = torch.optim.AdamW(
    param_list,
    lr=args.lr,
)
# %%
phase_ids = { phase: np.genfromtxt(f'files/{phase}_ids.txt') for phase in ['train', 'val', 'test'] }
if args.subsample is not None:
    phase_ids['train'] = phase_ids['train'][::args.subsample]
phase_ids['val'] = phase_ids['val'][::10]
datasets = { phase: utils.EHROutcomesDataset(
    args.ehr_outcomes_path,
    tokenizer,
    ids,
    max_length=512,
    code_resolution=args.code_resolution, # WARN: the benchmark may not parse to the same resolution
    shuffle_in_visit=phase=='train',
) for phase, ids in phase_ids.items() }

loaded_name = args.load.split('/')[-2]
scratchtag = ('pretrained_frozen-' if args.freeze else 'pretrained-') if not args.scratch else f'scratch-'
subtag = '' if args.subsample is None else f'sub{args.subsample}-'
mdlname = f'ft-{subtag}{scratchtag}{args.task}-ftlr{args.lr}-{loaded_name}'
print(mdlname)
training_args = TrainingArguments(
    output_dir=f'runs/{mdlname}',
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    eval_accumulation_steps=15,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    report_to='wandb' if not args.nowandb else None,
    evaluation_strategy='steps',
    run_name=mdlname,
    eval_steps=args.eval_steps,
    save_steps=2000,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pos_labels = labels[:, 1]
    predictions = np.argmax(logits, axis=-1)

    accuracy = (predictions == pos_labels).sum() / len(predictions)
    roc = roc_auc_score(pos_labels, predictions)
    ap = average_precision_score(pos_labels, predictions)
    f1 = f1_score(pos_labels > 0.5, predictions)

    return dict(
        accuracy=accuracy,
        pr=ap,
        roc=roc,
        f1=f1,
    )

class CustomCallback(TrainerCallback):
    def on_log(self, __args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            # any custom logging
            pass

if args.freeze:
    train_params = ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
    for name, param in model.named_parameters():
        if name not in train_params:
            param.requires_grad = False

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
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
torch.save(model.state_dict(), f'saved/{mdlname}.pth')
# %%
