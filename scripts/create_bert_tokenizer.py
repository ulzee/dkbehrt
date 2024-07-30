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
if not os.path.exists(f'{project_root}/saved/tokenizers'):
    os.mkdir(f'{project_root}/saved/tokenizers')
if not os.path.exists(f'{project_root}/saved/tokenizers/bert'):
    os.mkdir(f'{project_root}/saved/tokenizers/bert')
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
    '[PAD]',
    '[UNK]',
    '[CLS]',
    '[SEP]',
    '[MASK]',
]

new_token_set = required_tokens + [None for _ in vocab]

for word, vi in vocab.items():
    new_token_set[vi + len(required_tokens)] = word.lower()

with open(f'{project_root}/saved/tokenizers/medbert/vocab.txt', 'w') as fl:
    fl.write('\n'.join(new_token_set))
# %%
with open(glob(f'{project_root}/saved/models--google-bert--bert-base-uncased/snapshots/**/tokenizer.json')[0]) as fl:
    tkobj = json.load(fl)
tkobj
# %%
for tkn in tkobj['added_tokens']:
    tkn['id'] = new_token_set.index(tkn['content'])

tkobj['model']['vocab'] = { w: i for i, w in enumerate(new_token_set) }

tkobj['post_processor']['special_tokens']['[CLS]']['ids'] = [required_tokens.index('[CLS]')]
tkobj['post_processor']['special_tokens']['[SEP]']['ids'] = [required_tokens.index('[SEP]')]

tkobj
# %%
for fl in glob(f'{project_root}/saved/models--google-bert--bert-base-uncased/snapshots/**/*'):
    print(fl)
    shutil.copyfile(fl, f'{project_root}/saved/tokenizers/bert/' + fl.split('/')[-1])
# %%
with open(f'{project_root}/saved/tokenizers/bert/vocab.txt', 'w') as fl:
    fl.write('\n'.join(new_token_set))
# %%
with open(f'{project_root}/saved/tokenizers/bert/tokenizer.json', 'w') as fl:
    json.dump(tkobj, fl, indent=4)