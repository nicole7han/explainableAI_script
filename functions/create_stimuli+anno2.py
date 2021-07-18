import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
from scipy.stats import norm, multivariate_normal


def get_stimuli(lth, wth, ang, imageSizeX, imageSizeY):
    x = np.linspace(0, imageSizeX-1, imageSizeX).astype(int)
    y = np.linspace(0, imageSizeX-1, imageSizeY).astype(int)
    columnsInImage, rowsInImage = np.meshgrid(x, y)
    centerX = int((imageSizeX+1)/2)
    centerY = int((imageSizeY+1)/2)

    b = lth
    a = wth
    theta = (90-ang)*np.pi/180

    img = ( ( (columnsInImage - centerX)*np.cos(theta)+(rowsInImage - centerY)*np.sin(theta) )**2/a**2 + \
            ( ( columnsInImage - centerX)*np.sin(theta)-(rowsInImage - centerY)*np.cos(theta) )**2/b**2 <= 1).astype(float)\
          *140/255
    img[img==0]=.5
    sigma=.08
    stimuli = img + sigma*np.random.randn(img.shape[0],img.shape[1])

    return stimuli

N = 400 #number of stimuli

# create target
t_mean = [80,40,45]
t_cov = [[3,0,0],[0,3,0],[0,0,3]]
targets = np.random.multivariate_normal(t_mean, t_cov, N)
target_mins = np.min(targets, 0)
target_maxs = np.max(targets, 0)
targets_range = np.array([target_mins, target_maxs]).round()

# create distractor
d_mean = [78,42,43]
d_cov = [[3,0,0],[0,3,0],[0,0,3]]
distractors = np.random.multivariate_normal(d_mean, d_cov, N)
distractors_mins = np.min(distractors, 0)
distractors_maxs = np.max(distractors, 0)
distractors_range = np.array([distractors_mins, distractors_maxs]).round()

#visualize parameters in 3d
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
ax.scatter(targets[:,0], targets[:,1], targets[:,2], s=100, c='r')
ax.scatter(distractors[:,0], distractors[:,1], distractors[:,2], s=100, c='b')

# calculate all possible discrete targets (1200+) and distractors (1200+)
num=0
for f1 in np.arange(targets_range[0,0], targets_range[1,0]+1):
    for f2 in np.arange(targets_range[0,1], targets_range[1,1]+1):
        for f3 in np.arange(targets_range[0,2], targets_range[1,2]+1):
            num+=1
print("all possible discrete targets/distractors: {}".format(num))



imageSizeX = 255*3
imageSizeY = 255*3
stim_info = {}

n = 40 #number of samples for each block
for b in range(4):# range(int(N/n)):
    for i in range(n):
        lth, wdth, ang = targets[b*n+i,:].round()
        stimuli = get_stimuli(lth, wdth, ang, imageSizeX, imageSizeY)
        plt.imshow(stimuli, 'gray')
        plt.axis('off')
        plt.savefig('Stimuli/block{}/target{}.jpg'.format(b+1,n*b+i+1),  bbox_inches='tight', pad_inches=0)
        plt.close('all')
        stim_info['target{}'.format(n*b+i+1)] = {'length':lth, 'width':wdth, 'angle':ang, 'corr_ans':1}


        lth, wdth, ang = distractors[b*n+i,:].round()
        stimuli = get_stimuli(lth, wdth, ang, imageSizeX, imageSizeY)
        plt.imshow(stimuli, 'gray')
        plt.axis('off')
        plt.savefig('Stimuli/block{}/distractor{}.jpg'.format(b+1,n*b+i+1),  bbox_inches='tight', pad_inches=0)
        plt.close('all')

        # creating feedback for distractors (compare to targets)
        # pdf = multivariate_normal.pdf([lth, wdth, ang], mean=t_mean, cov=t_cov)
        # length
        # p_d = norm.cdf(lth, loc=d_mean[0], scale=d_cov[0][0]) # cdf of lth given distractor distribution
        p_t = norm.cdf(lth, loc=t_mean[0], scale=t_cov[0][0]) # cdf of lth given target distribution
        if p_t >= .7: # most target is smaller
            feedback = "This is a distractor. Most targets have shorter length, "
        elif p_t <= .3: # most target is smaller
            feedback = "This is a distractor. Most targets have longer length, "
        else: # around 50% target is
            feedback = "This is a distractor. Most targets have similar length, "

        # width
        p_t = norm.cdf(wdth, loc=t_mean[1], scale=t_cov[1][1])  # cdf of wdth given target distribution
        if p_t >= .7:
            feedback += "have smaller width, "
        elif p_t <= .3:
            feedback += "have larger width, "
        else:
            feedback += "have similar width, "

        # orientation
        p_t = norm.cdf(ang, loc=t_mean[2], scale=t_cov[2][2])  # cdf of angle given target distribution
        if p_t >= .7:
            feedback += "oriented more horizontally."
        elif p_t <= .3:
            feedback += "oriented more vertically."
        else:
            feedback += "with similar orientation."

        stim_info['distractor{}'.format(N*b+i+1)] = {'length':lth, 'width':wdth, 'angle':ang, 'corr_ans':0, 'feedback':feedback}

df = pd.DataFrame(stim_info).T
df = df.reset_index()
df.rename(columns={'index':'stim'}, inplace=True)
df.to_excel('Stimuli/stim_info.xlsx', index=None)