### implement ideal observer on target and distractor stimli ###
# feature1: length, distractor:[72,84], target:[74,86]
# feature2: width, distractor:[36,48], target:[34,46]
# feature2: angle, distractor:[37,49], target:[39,51]
import os.path

import pandas as pd
from scipy.special import logsumexp
import random
from functions.utils import *

aics = pd.read_excel('IO_data/aics.xlsx')
io_resp = pd.read_excel('IO_data/IO_resp.xlsx')

# organize all subjects data
cond = 'nfb' # with feedback condition or no feedback condition
data_path = 'Data/Data_{}'.format(cond)
subjects = glob.glob('{}/*'.format(data_path))
allresp = pd.DataFrame()
for subj in subjects:
    subj_name = os.path.split(subj)[-1]
    resp = organize_humanresp(data_path, subj_name)
    resp['subject'] = subj_name[4:]
    resp['correct'] = resp.apply(label_correct, axis=1)
    allresp = allresp.append(resp, ignore_index=True)
allresp.to_excel('Data/Human_resp_{}.xlsx'.format(cond))


# PC over time
subj_mean = allresp.groupby(['subject','block_num']).mean().reset_index()
subj_mean['correct'] = subj_mean['correct']*100
sns_setup_small(sns)
ax = sns.lineplot(x = 'block_num', y ='correct', data = subj_mean, hue='subject')
ax.set(xlabel='block number', ylabel='percentage correct (%)',
       ylim= [40,80])
ax.figure.savefig("Figures_{}/PC_vs_block_{}.png".format(cond, cond))
plt.close()


# Confidence over time
sns_setup_small(sns)
ax = sns.lineplot(x = 'block_num', y = 'conf', data = subj_mean, hue='subject')
ax.set(xlabel='block number', ylabel='confidence level',
       ylim= [1,5])
ax.figure.savefig("Figures_{}/conf_vs_block.png".format(cond))
plt.close()



# percentage of responding target
subjects = np.unique(allresp['subject'])
features = ['length','width','angle']
fig, axes = plt.subplots(nrows=len(features), ncols=7, figsize=(15, 10))
sns_setup_small(sns)

for i,f in enumerate(features):
    # f_ptrials = allresp.groupby([ f]).mean().reset_index()
    # sns.lineplot(x=f_ptrials[f], y=f_ptrials['gt'], ax=axes[i, 0])
    # if i==0:
    #     axes[i,0].set_title("ground truth")

    # io_ptrials = io_resp.groupby([f]).mean().reset_index()
    # sns.lineplot(x=io_ptrials[f], y=io_ptrials['gt'], ax=axes[i, 0])
    if i==0:
        axes[i,0].set_title("IO")
    axes[i,0].set(ylabel='% target trials ({})'.format(f))

    for b in range(6):
        # try:
        #     aic = round(aics[(aics['block_num']==(b+1)) & (aics['feature']==f)]['aic'].item(),1)
        # except: pass
        b_resp = allresp[allresp['block_num'] == (b + 1)]
        ptrials = b_resp.groupby(['subject', f]).mean().reset_index()
        sns.lineplot(x=ptrials[f], y=ptrials['resp'], ax=axes[i, b+1])
        axes[i, b+1].set(xlabel='',ylabel='')
        # try:
        #     axes[i, b+1].text(0.1, 0.9, 'aic={}'.format(aic), fontsize=13, transform=axes[i, b+1].transAxes)
        # except: pass
        axes[i, b+1].xaxis.set_visible(False)
        axes[i, b+1].yaxis.set_visible(False)
        if i == 0:
            axes[i, b+1].set_title("block {}".format(b+1))

# plt.tight_layout()
plt.savefig('Figures_nfb/tuningcurve_subjmean.jpg')
plt.close(fig)


# confidence level
features = ['length','width','angle']
fig, axes = plt.subplots(nrows=len(features), ncols=7, figsize=(15, 10))
sns_setup_small(sns)
for i,f in enumerate(features):
    f_ptrials = allresp.groupby([ f]).mean().reset_index()
    sns.lineplot(x=f_ptrials[f], y=f_ptrials['gt'], ax=axes[i, 0])
    if i==0:
        axes[i,0].set_title("ground truth")
    axes[i,0].set(ylabel='% target trials ({})'.format(f))

    for b in range(6):
        b_resp = allresp[allresp['block_num'] == (b + 1)]
        ptrials = b_resp.groupby(['subject', f]).mean().reset_index()
        sns.lineplot(x=ptrials[f], y=ptrials['conf'], ax=axes[i, b+1])
        axes[i, b+1].xaxis.set_visible(False)
        if i == 0:
            axes[i, b+1].set_title("block {}".format(b+1))
        if b == 0:
            axes[i, b + 1].set(xlabel='', ylabel='confidence')
        else:
            axes[i, b + 1].yaxis.set_visible(False)
            axes[i, b + 1].set(xlabel='', ylabel='')
plt.savefig('Figures/confidence_subjmean.jpg')
plt.close(fig)


# # Tuning Curve
# for b in range(6):
#     b_resp = allresp[allresp['block_num'] == (b + 1)]
#     for subj in subjects:
#         subj_resp = b_resp[b_resp['subject']==subj]
#         ptrials = prop_resp(b_resp)
#
#     plot_tuningcurve(b_resp, '{}_block_{}'.format(subject, b + 1))
#     b_resp['correct'] = b_resp.apply(label_correct, axis=1)
#     print("PC = {}".format(b_resp['correct'].mean()))
