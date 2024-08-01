#%%
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import numpy as np
#%%
tokenizer = AutoTokenizer.from_pretrained('../saved/tokenizers/medbert')
# %%
w = torch.load('../saved/medbert_basic.pth', map_location='cpu')
# w = torch.load('../saved/medbert_basic_0.pth', map_location='cpu')
# %%
w.keys()
# %%
t = w['bert.embeddings.word_embeddings.weight']
t.shape
#%%
emat = t.numpy()
emat -= np.mean(emat, 0)
emat /= np.std(emat, 0)
# %%
# xs = PCA(n_components=2).fit_transform(emat)
xs = TSNE(n_components=2).fit_transform(emat)
xs.shape
# %%
alph = 'abcdefghijkl'
plt.figure()
for aa in alph:
    inds = [ti for tkn, ti in tokenizer.vocab.items() if aa in tkn]
    plt.scatter(*xs[inds, :2].T, label=aa)
plt.legend()
plt.show()
# %%
