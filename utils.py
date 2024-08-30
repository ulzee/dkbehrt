
import torch
from torch.utils.data import Dataset
import numpy as np
from random import shuffle
import pickle as pk
import pandas as pd
from tqdm import tqdm

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
    def __init__(self, dxs, tokenizer, patient_ids, covs, separator, max_length=None, shuffle_in_visit=True):
        self.dxs = dxs
        self.shuffle_in_visit = shuffle_in_visit
        self.max_length = max_length
        self.covs = covs
        self.tokenizer = tokenizer
        self.separator = separator
        pdict = { i: True for i in patient_ids }
        self.pids = [i for i in dxs if i in pdict]

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        concepts = self.dxs[self.pids[i]]['codes']['concepts']
        visits = self.dxs[self.pids[i]]['codes']['visits']
        vid = visits[0]

        byvisit = []
        cinv = []
        current_v = visits[0]
        for c, v in zip(concepts, visits):
            if v != current_v:
                byvisit += [(current_v, cinv)]
                current_v = v
                cinv = []
            cinv += [c]
        if len(cinv):
            byvisit += [(current_v, cinv)]

        code_series = ''
        if self.covs is not None: cov_hist = []
        for vi, (vid, cinv) in enumerate(byvisit):
            if self.shuffle_in_visit:
                shuffle(cinv)
            for c in cinv:
                code_series += f' {c}'
                if self.covs is not None: cov_hist += [self.covs[vid]]
            if vi != len(byvisit) - 1:
                code_series += f' {self.separator}'
                if self.covs is not None: cov_hist += [self.covs[None]]

        sample = { k: v for k, v in self.tokenizer(
            code_series, padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length).items()  if k not in ['token_type_ids']}

        if self.covs is not None:
            blank_cov = [self.covs[None]]
            if self.max_length is not None:
                cov_hist = cov_hist[:self.max_length-2]
            sample['position_ids'] = blank_cov + cov_hist + blank_cov # matches tokenizer pad

        return sample

class EHROutcomesDataset(Dataset):
    def __init__(self, task, ehr_outcomes_path, tokenizer, patient_ids, covs=None, code_resolution=5, separator='[SEP]', max_length=None, shuffle_in_visit=True, verbose=True):
        with open(f'{ehr_outcomes_path}/dx.pk', 'rb') as fl:
            self.dxs = pk.load(fl)

        if task in ['mortality', 'los72']:
            self.stays = pd.read_csv(f'{ehr_outcomes_path}/targets_by_icustay.csv')
        else:
            self.stays = pd.read_csv(f'{ehr_outcomes_path}/targets_diagnosis_{task}.csv')

        self.shuffle_in_visit = shuffle_in_visit
        self.tokenizer = tokenizer
        self.separator = separator
        self.code_resolution = code_resolution
        self.max_length = max_length
        self.covs = covs
        self.task = task

        pdict = { i: True for i in patient_ids }
        self.pids = [i for i in self.stays['subject_id'].unique() if i in pdict]
        self.stays = self.stays.set_index('subject_id').loc[self.pids].reset_index()
        self.history = [
            [[c[:code_resolution].lower() for c in self.dxs[h] if type(c) == str] if h in self.dxs else [] \
                for h in eval(hids)] \
                    for hids in (tqdm if verbose else lambda x: x)(self.stays['past_visits'])]
        self.labels = self.stays[task].values.tolist()

    def __len__(self):
        return len(self.stays)

    def __getitem__(self, i):

        code_series = ''
        byvisit = self.history[i]
        visit_ids = eval(self.stays['past_visits'][i])
        if self.covs is not None: cov_hist = []
        for vi, (vid, cinv) in enumerate(zip(visit_ids, byvisit)):
            if self.shuffle_in_visit:
                shuffle(cinv)
            for c in cinv:
                code_series += f' {c}'
                if self.covs is not None: cov_hist += [self.covs[vid]]
            if vi != len(byvisit) - 1:
                code_series += f' {self.separator}'
                if self.covs is not None: cov_hist += [self.covs[None]]

        sample = { k: v for k, v in self.tokenizer(
            code_series, padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length).items()  if k not in ['token_type_ids']}
        sample['labels'] = [[1.0, 0.0], [0.0, 1.0]][self.labels[i]]

        if self.covs is not None:
            blank_cov = [self.covs[None]]
            if self.max_length is not None:
                cov_hist = cov_hist[:self.max_length-2]
            sample['position_ids'] = blank_cov + cov_hist + blank_cov # matches tokenizer pad

        return sample

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

def load_covariates(covfile='saved/cov.csv', covlist=['gender', 'age']):
    cdf = pd.read_csv(covfile)
    covs = dict()
    def format_covs_as_position(covls):
        flip_factor = 1
        pos_value = 0.01
        for cv, cname in zip(covls, covlist):
            if cname == 'gender':
                if cv in 'MF':
                    flip_factor = [-1, 1]['MF'.index(cv)]
                else:
                    raise Exception('Not supported')
            elif cname == 'age':
                pos_value = cv/100/100
            else:
                print('WARN: unknown covariate')
        return flip_factor * pos_value

    for sample_covs in zip(*([cdf['hadm_id']] + [cdf[c] for c in covlist])):
        sid, use_covs = sample_covs[0], sample_covs[1:]
        covs[sid] = format_covs_as_position(use_covs)
    covs[None] = 0
    return covs