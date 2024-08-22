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
parser.add_argument('--batch_size', type=int, default=192)
parser.add_argument('--eval_batch_size', type=int, default=64)
parser.add_argument('--eval_steps', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--disable_visible_devices', action='store_true', default=False)
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--ehr_outcomes_path', type=str, default='../ehr-outcomes')
parser.add_argument('--nowandb', action='store_true', default=False)
parser.add_argument('--predict', action='store_true', default=False)
parser.add_argument('--predict_set', default='test', type=str)
parser.add_argument('--val_subsample', default=10, type=int)
parser.add_argument('--eval_test', default=False, action='store_true')
args = parser.parse_args()
#%%
if not args.nowandb:
    os.environ["WANDB_PROJECT"] = f'ehr-outcomes-{args.task}'
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
model_mode = args.load.split('/')[-2].split('-')[1] # expects model_name/checkpoint_num folders
#%%
with open(f'saved/diagnoses-cr{args.code_resolution}.pk', 'rb') as fl:
    dxs = pk.load(fl)
#%%
tokenizer = AutoTokenizer.from_pretrained(f'./saved/tokenizers/bert-cr{args.code_resolution}')
#%%
loaded_name = args.load.split('/')[-2]
scratchtag = ('pretrained_frozen-' if args.freeze else 'pretrained-') if not args.scratch else f'scratch-'
subtag = '' if args.subsample is None else f'sub{args.subsample}-'
mdlname = f'ft-{subtag}{scratchtag}{args.task}-ftlr{args.lr}-{loaded_name}'
print(mdlname)

model = BertForSequenceClassification.from_pretrained(args.load, num_labels=2)
if model_mode == 'base':
    if args.scratch:
        model = BertForSequenceClassification(model.config)
elif model_mode == 'attn':

    # load embeddings
    # use_embedding = '../data/icd10/embeddings/kane/biogpt100_collated.pk'
    use_embedding = '../data/icd10/embeddings/hug/microsoft--biogpt_hidden_collated.pk'
    with open(use_embedding, 'rb') as fl:
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

    # reset the model for now
    model = BertForSequenceClassification(model.config)
    bertconfig = model.config
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

    if not args.scratch:
        model.load_state_dict(torch.load(f'saved/{loaded_name}.pth'), strict=False)
        # NOTE: until we can figure out a way to make hug trainer save custom models, we need to load from a state dict
#%%
phase_ids = { phase: np.genfromtxt(f'files/{phase}_ids.txt') for phase in ['train', 'val', 'test'] }
if args.subsample is not None:
    phase_ids['train'] = phase_ids['train'][::args.subsample]
if not args.predict:
    # reduce val set to not slow down training
    phase_ids['val'] = phase_ids['val'][::args.val_subsample]



val_ids = phase_ids['val']
test_ids = phase_ids['test']
phase_ids['val'] = val_ids[:len(val_ids)//2].tolist() + test_ids[:len(test_ids)//2].tolist()
phase_ids['test'] = val_ids[len(val_ids)//2:].tolist() + test_ids[len(test_ids)//2:].tolist()

datasets = { phase: utils.EHROutcomesDataset(
    args.task,
    args.ehr_outcomes_path,
    tokenizer,
    ids,
    max_length=512,
    code_resolution=args.code_resolution, # WARN: the benchmark may not parse to the same resolution
    shuffle_in_visit=phase=='train',
) for phase, ids in phase_ids.items() }

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
            if 'eval_loss' in logs:
                if self.best_loss is None or self.best_loss > logs['eval_loss']:
                    self.best_loss = logs['eval_loss']

                    print('saved ckpt:', logs['eval_loss'])
                    torch.save(model.state_dict(), f'saved/best_{mdlname}.pth')

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
        with open(f'../ehr-outcomes/files/boot_ixs.pk', 'rb') as fl:
            boot_ixs = pk.load(fl)
    else:
        with open(f'../ehr-outcomes/files/boot_ixs_{args.task}.pk', 'rb') as fl:
            boot_ixs = pk.load(fl)

    def get_boot_metrics(ypred):
        ls = []
        ytrue = np.array(datasets[args.predict_set].labels)
        for bxs in boot_ixs:
            ls += [[
                average_precision_score(ytrue[bxs], ypred[bxs]),
                roc_auc_score(ytrue[bxs], ypred[bxs]),
                f1_score(ytrue[bxs], ypred[bxs] > 0.5, average='macro'),
            ]]

        out = ''
        for tag, replicates in zip(['ap', 'roc', 'f1'], zip(*ls)):
            est = np.mean(replicates)
            ci = 1.95*np.std(replicates)
            out += f'{tag}:{est:.3f} ({ci:.3f}) '

        print(out)


    model.load_state_dict(torch.load(f'saved/best_{mdlname}.pth'))
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
