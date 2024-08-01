#%%
import os, sys
sys.path.append('.')
sys.path.append('..')
import pandas as pd
from paths import mimic_root, project_root
import numpy as np
from icdmappings import Mapper
# %%
icddf = pd.read_csv(f'{mimic_root}/hosp/diagnoses_icd.csv')
# %%
icd9df = icddf[icddf['icd_version'] == 9]
icd9s = icd9df['icd_code']
icd9s
#%%
codes = icddf['icd_code'].values
versions = icddf['icd_version'].values
# %%
mapper = Mapper()
icd9s_to_10s = [mapper.map(c, source='icd9', target='icd10') if v == 9 else c for c, v in zip(codes, versions)]
icd9s_to_10s = [None if c == 'NoDx' else c for c in icd9s_to_10s]
len(icd9s_to_10s), len([c for c in icd9s_to_10s if c == None])
# %%
icddf['icd10'] = icd9s_to_10s
icddf
#%%
len([c for c in icddf[icddf['icd_version'] == 9]['icd10'].values if c is not None and c[0] == 'F'])
# %%
icddf.to_csv(f'{project_root}/saved/hosp-diagnoses_icd.csv', index=False)
# %%
