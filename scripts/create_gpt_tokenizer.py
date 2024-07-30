#%%
import os, sys
sys.path.append('.')
sys.path.append('..')
from transformers import AutoTokenizer
from paths import project_root
from huggingface_hub import hf_hub_download
import pickle as pk
import shutil
from glob import glob
import json
#%%
hf_hub_download(repo_id='google-bert/bert-base-uncased', filename='config.json', cache_dir='saved/')
hf_hub_download(repo_id='google-bert/bert-base-uncased', filename='tokenizer.json', cache_dir='saved/')
hf_hub_download(repo_id='google-bert/bert-base-uncased', filename='tokenizer_config.json', cache_dir='saved/')
hf_hub_download(repo_id='google-bert/bert-base-uncased', filename='vocab.txt', cache_dir='saved/')
#%%
with open('saved/vocab.pk', 'rb') as fl:
    vocab = pk.load(fl)
len(vocab)
# %%
required_tokens = [
    '<unk>',
    '<s>',
    '</s>',
    '<v>'
]

new_token_set = required_tokens + [None for _ in vocab]

for word, vi in vocab.items():
    new_token_set[vi + len(required_tokens)] = word.lower()
# %%
# with open(glob(f'{project_root}/saved/models--mistralai--Mistral-7B-v0.1/snapshots/**/tokenizer.json')[0]) as fl:
with open(glob(f'{project_root}/saved/models--google-bert--bert-base-uncased/snapshots/**/tokenizer.json')[0]) as fl:
    tkobj = json.load(fl)
tkobj
# %%
tkobj['added_tokens'] = tkobj['added_tokens'][:len(required_tokens)]
for i, tkn in enumerate(tkobj['added_tokens']):
    tkn['id'] = i
    tkn['content'] = required_tokens[i]

tkobj['model']['unk_token'] = '<unk>'
tkobj['model']['vocab'] = { w: i for i, w in enumerate(new_token_set) }
# tkobj['model']['merges'] = []

tkobj
#%%
tkobj['post_processor']['single'] = [
    {
        "SpecialToken": {
            "id": "<s>",
            "type_id": 0
        }
    },
    {
        "Sequence": {
            "id": "A",
            "type_id": 0
        }
    }
]
tkobj['post_processor']['special_tokens'] = {
    "<s>": {
        "id": "<s>",
        "ids": [
            1
        ],
        "tokens": [
            "<s>"
        ]
    },
    "</s>": {
        "id": "</s>",
        "ids": [
            2
        ],
        "tokens": [
            "</s>"
        ]
    }
}
#%%
# for fl in glob(f'{project_root}/saved/models--mistralai--Mistral-7B-v0.1/snapshots/**/*.json'):
# for fl in glob(f'{project_root}/saved/models--google-bert--bert-base-uncased/snapshots/**/*'):
for fl in glob(f'{project_root}/saved/models--google-bert--bert-base-uncased/snapshots/**/*'):
    print(fl)
    shutil.copyfile(fl, f'{project_root}/saved/tokenizers/gpt/' + fl.split('/')[-1])
#%%
with open(f'{project_root}/saved/tokenizers/gpt/vocab.txt', 'w') as fl:
    fl.write('\n'.join(new_token_set))
# %%
with open(f'{project_root}/saved/tokenizers/gpt/tokenizer.json', 'w') as fl:
    json.dump(tkobj, fl, indent=4)
#%%