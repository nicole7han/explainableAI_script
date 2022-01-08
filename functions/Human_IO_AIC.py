import pandas as pd
from functions.utils import *
from scipy.stats import norm

human_resp = pd.read_excel('Data/Human_resp_fb.xlsx')
io_resp = pd.read_excel('IO_data/IO_resp.xlsx')

# compute AIC: using human tuning curve to fit IO
# AIC = -2(log-likelihood) + 2K
features = ['length','width','angle']
aics = pd.DataFrame()
for i,f in enumerate(features):
    for b in range(6):
        b_resp = human_resp[human_resp['block_num'] == (b + 1)]
        h_ptrials = b_resp.groupby(['subject', f]).mean().reset_index()
        h_ptrials_sd = h_ptrials.groupby([f]).std().reset_index()
        h_ptrials = h_ptrials.groupby([f]).mean().reset_index()
        io_ptrials = io_resp.groupby([f]).mean().reset_index()
        h_ptrials['io_resp'] = io_ptrials['resp']
        h_ptrials['resp_sd'] = h_ptrials_sd['resp']
        h_ptrials['logl'] = np.log(norm(h_ptrials['resp'], h_ptrials['resp_sd']).pdf(h_ptrials['io_resp']))
        aic = -2*h_ptrials['logl'].sum() + 2*2
        aics = aics.append({'feature':f, 'aic': aic, 'block_num':b+1}, ignore_index=True)
aics.to_excel('IO_data/aics.xlsx')