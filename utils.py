
import torch
from torch.utils.data import Dataset
import numpy as np
from random import shuffle

def gather_hidden_params(mdl):

    c = 0
    for ch in mdl.children():
        c += gather_hidden_params(ch)
    return c

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params += gather_hidden_params(model)

    return trainable_params

def paramscan(mdl, d=0):

    for ch in mdl.children():
        print(' '*d, count_parameters(ch), str(ch)[:10])
        paramscan(ch, d=d+1)

mf = lambda c: f'{c/1000/1000:.1f}M'
bf = lambda c: f'{c/1000/1000/1000:.1f}B'

class ICDDataset(Dataset):
    def __init__(self, dxs, tokenizer, patient_ids, separator, max_length=None, shuffle_in_visit=True):
        self.dxs = dxs
        self.shuffle_in_visit = shuffle_in_visit
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.separator = separator
        pdict = { i: True for i in patient_ids }
        self.pids = [i for i in dxs if i in pdict]
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, i):
        # FIXME: code cutoff id diagnoses file?
        concepts = self.dxs[self.pids[i]]['codes']['concepts']
        visits = self.dxs[self.pids[i]]['codes']['visits']
        vid = visits[0]

        byvisit = []
        cinv = []
        vid = visits[0]
        for c, v in zip(concepts, visits):
            if v != vid:
                vid = v
                byvisit += [cinv]
                cinv = []
            cinv += [c]
        if len(cinv):
            byvisit += [cinv]

        code_series = ''
        for vi, cinv in enumerate(byvisit):
            if self.shuffle_in_visit:
                shuffle(cinv)
            for c in cinv:
                code_series += f' {c}'
            if vi != len(byvisit) - 1: code_series += f' {self.separator}'

        return { k: v for k, v in self.tokenizer(
            code_series, padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length).items()  if k not in ['token_type_ids']}

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class HugMetrics:

    class TopN:
        def __call__(self, eval_pred, mask_value=-100, topns=(1, 5, 10)):
            logits, labels = eval_pred
            bsize, seqlen = labels.shape

            logits = torch.from_numpy(np.reshape(logits, (bsize*seqlen, -1)))
            labels = torch.from_numpy(np.reshape(labels, (bsize*seqlen)))
            where_prediction = labels != mask_value

            topaccs = topk_accuracy(logits[where_prediction], labels[where_prediction], topk=topns)

            assert { n: acc for n, acc in zip(topns, topaccs) }