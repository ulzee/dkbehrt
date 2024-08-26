#%%
import os, sys
sys.path.append('.')
sys.path.append('..')
from paths import project_root, ukbb_root
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pk
import json
from dateutil.relativedelta import relativedelta
#%%
code_resolution = int(sys.argv[1])
#%%
pdf = pd.read_csv(f'{ukbb_root}/omop/omop_person.txt', usecols=['eid', 'gender_concept_id', 'year_of_birth', 'month_of_birth'], sep='\t')
pdf.columns = ['subject_id', 'gender_concept_id', 'byear', 'bmonth']
pdf['gender'] = [['M', 'F'][v == 8532] for v in pdf['gender_concept_id']]
pdf['birthdate'] = [datetime.strptime(f'{y}-{m}', '%Y-%m') for y, m in zip(pdf['byear'], pdf['bmonth'])]
pdf = pdf[['subject_id', 'gender', 'birthdate']]
pdf
# %%
obsdf = pd.read_csv(f'{ukbb_root}/omop/omop_condition_occurrence.txt', usecols=['eid', 'visit_occurrence_id', 'condition_type_concept_id', 'condition_source_concept_id', 'condition_source_value'], sep='\t')
obsdf
# %%
icddf = obsdf[(obsdf['condition_type_concept_id'] == 32817) & ~obsdf['visit_occurrence_id'].isna()]
icddf.columns = ['subject_id', 'condition_type_concept_id', 'hadm_id', 'icd10', 'condition_source_concept_id']
icddf['hadm_id'] = icddf['hadm_id'].astype(int)
icddf
#%%
hadmdf = pd.read_csv(f'{ukbb_root}/omop/omop_visit_occurrence.txt', usecols=['eid', 'visit_occurrence_id', 'visit_start_date', 'visit_end_date'], sep='\t')
hadmdf.columns = ['subject_id', 'hadm_id', 'admittime', 'dischtime']
hadmdf['admittime_dt'] = hadmdf['admittime'].map(lambda y: datetime.strptime(str(y), "%d/%m/%Y"))
hadmdf
# %%
hicd10df = hadmdf.set_index('hadm_id')[['admittime']].join(icddf.set_index('hadm_id'), how='right')
hicd10df
#%%
phicd10df = hicd10df.reset_index().set_index('subject_id').join(pdf.set_index('subject_id'), on='subject_id', how='left')
phicd10df = phicd10df[~phicd10df['icd10'].isna()]
phicd10df
#%%
phicd10df['admittime_dt'] = phicd10df['admittime'].map(lambda y: datetime.strptime(str(y), "%d/%m/%Y"))
phicd10df
#%%
# phicd10df['tsince'] = (phicd10df['admittime_dt'] - phicd10df['birthdate']).dt.total_seconds()
# %%
histdf = phicd10df.reset_index().sort_values('admittime_dt')[['subject_id', 'icd10', 'admittime_dt', 'hadm_id']].groupby('subject_id').agg(list)
histdf
#%%
lsd = { d: histdf[d].values for d in ['icd10', 'admittime_dt', 'hadm_id'] }
# %%
born_time = { pid: birthdate for pid, birthdate in zip(
    phicd10df.index, phicd10df['birthdate'].values) }
sex = { pid: s for pid, s in zip(pdf['subject_id'].values, pdf['gender'].values) }
#%%
format_code = lambda c: c[:code_resolution] if code_resolution is not None else c
vocab = dict()
bypatient = dict()
for i in tqdm(range(len(lsd['hadm_id']))):
    patient = histdf.index[i]
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
len(vocab)
# %%
with open(f'{project_root}/saved/ukbb/diagnoses-cr{code_resolution}.pk', 'wb') as fl:
    pk.dump(bypatient, fl)
# %%
with open(f'{project_root}/saved/ukbb/vocab-cr{code_resolution}.pk', 'wb') as fl:
    pk.dump(vocab, fl)
# %%
covdf = hadmdf.set_index('subject_id').join(pdf.set_index('subject_id'), on='subject_id', how='left').reset_index()
covdf['age'] = (covdf['admittime_dt'] - covdf['birthdate']).dt.total_seconds()/60/60/24/365
covdf_save = covdf[['subject_id', 'hadm_id', 'gender', 'age']]
covdf_save
# %%
covdf_save.to_csv(f'{project_root}/saved/ukbb/cov.csv', index=False)
# %%

