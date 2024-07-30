#%%
import os, sys
sys.path.append('.')
sys.path.append('..')
from paths import project_root, mimic_root
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pk
import json
from dateutil.relativedelta import relativedelta
#%%
code_resolution = 3
#%%
pdf = pd.read_csv(f'{mimic_root}/hosp/patients.csv')
pdf
# %%
icddf = pd.read_csv(f'{project_root}/saved/hosp-diagnoses_icd.csv')
icddf
#%%
hadmdf = pd.read_csv(f'{mimic_root}/hosp/admissions.csv')
hadmdf
# %%
hicd10df = hadmdf.set_index('hadm_id')[['admittime', 'dischtime']].join(icddf.set_index('hadm_id'), how='right')
hicd10df
#%%
phicd10df = hicd10df.reset_index().set_index('subject_id').join(pdf.set_index('subject_id'), on='subject_id', how='left')
phicd10df = phicd10df[~phicd10df['icd10'].isna()]
phicd10df
#%%
phicd10df['admittime_dt'] = phicd10df['admittime'].map(lambda y: datetime.strptime(str(y), "%Y-%m-%d %H:%M:%S"))
phicd10df['anchor_year_dt'] = phicd10df['anchor_year'].map(lambda y: datetime.strptime(str(y), "%Y"))
phicd10df
#%%
phicd10df['tsince'] = (phicd10df['admittime_dt'] - phicd10df['anchor_year_dt']).dt.total_seconds()
# %%
# roughly times since their first visit where they visited again
plt.figure()
plt.hist([
    phicd10df[phicd10df['icd_version'] == 9]['tsince'],
    phicd10df[phicd10df['icd_version'] == 10]['tsince'],
])
plt.show()
# %%
# histdf = phicd10df[phicd10df['icd_version'] == 10].reset_index().sort_values('admittime_dt')[['subject_id', 'icd10', 'admittime_dt', 'hadm_id', 'icd_version']].groupby('subject_id').agg(list)
histdf = phicd10df.reset_index().sort_values('admittime_dt')[['subject_id', 'icd10', 'admittime_dt', 'hadm_id', 'icd_version']].groupby('subject_id').agg(list)
histdf
#%%
lsd = { d: histdf[d].values for d in ['icd10', 'admittime_dt', 'hadm_id', 'icd_version'] }
# %%
born_time = { pid: y - np.timedelta64(365*age, 'D') for pid, age, y in zip(
    phicd10df.index, phicd10df['anchor_age'].values, phicd10df['anchor_year_dt'].values) }
#%%
sex = { pid: s for pid, s in zip(pdf['subject_id'].values, pdf['gender'].values) }
#%%
format_code = lambda c: c[:code_resolution] if code_resolution is not None else c
vocab = dict()
bypatient = dict()
for i in tqdm(range(len(lsd['hadm_id']))):
    patient = histdf.index[i]
    # icds, adts, adids, vers = [np.array(lsd[k][i]) for k in ['icd10', 'admittime_dt', 'hadm_id', 'icd_version']]
    name_remap = dict(
        icd10='concepts',
        admittime_dt='abspos',
        hadm_id='visits'
    )
    bypatient[patient] = dict(
        born=born_time[patient],
        sex=sex[patient],
        codes={
            (name_remap[k] if k in name_remap else k): lsd[k][i] for k in ['icd10', 'admittime_dt', 'hadm_id']
        }
    )
    bypatient[patient]['codes']['concepts'] = [format_code(c) for c in bypatient[patient]['codes']['concepts']]
    for c in lsd['icd10'][i]:
        c = format_code(c)
        if c not in vocab:
            vocab[c] = len(vocab)
    # break
# %%
with open(f'{project_root}/saved/diagnoses.pk', 'wb') as fl:
    pk.dump(bypatient, fl)
# %%
with open(f'{project_root}/saved/vocab.pk', 'wb') as fl:
    pk.dump(vocab, fl)
# %%