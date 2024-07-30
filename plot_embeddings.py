#%%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer
#%%
tokenizer = AutoTokenizer.from_pretrained('./saved/tokenizers/gpt')
# %%
d = torch.load('saved/gpt_basic.pth', map_location='cpu')
d.keys()
# %%
emat = d['model.embed_tokens.weight'].numpy()
# %%
xs = TSNE(n_components=2, random_state=0).fit_transform(emat)
# %%
alph = 'abcdefghilkl'
plt.figure()
for aa in alph:
    code_ixs = [i for w, i in tokenizer.vocab.items() if w[0] == aa]
    plt.scatter(*xs[code_ixs].T, label=aa)
plt.legend()
plt.show()
# %%
