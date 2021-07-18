import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd

def get_stimuli(s, ar, ang, imageSizeX, imageSizeY):
    x = np.linspace(0, imageSizeX-1, imageSizeX).astype(int)
    y = np.linspace(0, imageSizeX-1, imageSizeY).astype(int)
    columnsInImage, rowsInImage = np.meshgrid(x, y)
    centerX = int((imageSizeX+1)/2)
    centerY = int((imageSizeY+1)/2)

    b = s
    a = b * ar
    theta = (180-ang)*np.pi/180

    img = ( ( (columnsInImage - centerX)*np.cos(theta)+(rowsInImage - centerY)*np.sin(theta) )**2/a**2 + \
            ( ( columnsInImage - centerX)*np.sin(theta)-(rowsInImage - centerY)*np.cos(theta) )**2/b**2 <= 1).astype(float)\
          *140/255
    img[img==0]=.5
    sigma=.08
    stimuli = img + sigma*np.random.randn(img.shape[0],img.shape[1])

    return stimuli


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


# phi = np.linspace(0, np.pi, 20)
# theta = np.linspace(0, 2 * np.pi, 40)
# x = np.outer(np.sin(theta), np.cos(phi))
# y = np.outer(np.sin(theta), np.sin(phi))
# z = np.outer(np.cos(theta), np.ones_like(phi))

# create stimuli with three features:
# feature1: length
# feature2: width
# feature3: orientation

N = 50
x_t, y_t, z_t = sample_spherical(N) #sample on a sphere surface
z_t *= 3 #increase the range of third dimension, otherwise there's not enough variance

x_d, y_d, z_d = sample_spherical(N) #sample on a sphere surface
x_d = x_t * 5
y_d = y_t * 1.5
z_d = z_t * 4

targets = np.array([x_t+30, y_t+8, z_t+45]) # move to the mean
distractors = np.array([x_d+30, y_d+8, z_d+45])
target_mins = np.min(targets, 1)
target_maxs = np.max(targets, 1)
targets_range = np.array([target_mins, target_maxs])

distractors_mins = np.min(distractors, 1)
distractors_maxs = np.max(distractors, 1)
distractors_range = np.array([distractors_mins, distractors_maxs])

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
ax.scatter(targets[0,:], targets[1,:], targets[2,:], s=100, c='r')
ax.scatter(distractors[0,:], distractors[1,:], distractors[2,:], s=100, c='b')


imageSizeX = 255*3
imageSizeY = 255*3
stim_info = {}

N = 20 #number of samples for each block
for b in range(4):
    os.makedirs('Stimuli/block{}'.format(b+1), exist_ok=True)

    x_t, y_t, z_t = sample_spherical(N)  # sample on a sphere surface
    z_t *= 3  # increase the range of third dimension, otherwise there's not enough variance

    x_d, y_d, z_d = sample_spherical(N)  # sample on a sphere surface
    x_d = x_t * 5
    y_d = y_t * 1.5
    z_d = z_t * 4

    targets = np.array([x_t + 30, y_t + 8, z_t + 45])
    distractors = np.array([x_d + 30, y_d + 8, z_d + 45])

    for i in range(N):
        s, ar, ang = targets[:,i]
        s = round(s)
        stimuli = get_stimuli(s, ar, ang, imageSizeX, imageSizeY)
        plt.imshow(stimuli, 'gray')
        plt.axis('off')
        plt.savefig('Stimuli/block{}/target{}.jpg'.format(b+1,N*b+i+1),  bbox_inches='tight', pad_inches=0)
        plt.close('all')
        stim_info['target{}'.format(N*b+i+1)] = {'size':s, 'aspect_r':ar, 'angle':ang, 'corr_ans':1}


        s, ar, ang = distractors[:,i]
        s = round(s)
        stimuli = get_stimuli(s, ar, ang, imageSizeX, imageSizeY)
        plt.imshow(stimuli, 'gray')
        plt.axis('off')
        plt.savefig('Stimuli/block{}/distractor{}.jpg'.format(b+1,N*b+i+1), bbox_inches='tight', pad_inches=0)
        plt.close('all')

        # size
        if target_maxs[0] < s: # target is smaller
            feedback = "This is a distractor. Target should be smaller in size, "
        elif target_mins[0] > s: # target is larger
            feedback = "This is a distractor. Target should be larger in size, "
        else: # size i
            feedback = "This is a distractor. Target should be with similar size, "

        # aspect ratio
        if target_maxs[1] < ar:
            feedback += "with smaller ratio along two axes, "
        elif target_mins[1] > ar:
            feedback += "with larger ratio along two axes, "
        else:
            feedback += "with similar ratio along two axes, "

        # orientation
        if target_maxs[2] < ang:
            feedback += "and more oriented horizontally."
        elif target_mins[2] > ang:
            feedback += "and more oriented vertically."
        else:
            feedback += "and with similar orientation."

        stim_info['distractor{}'.format(N*b+i+1)] = {'size':s, 'aspect_r':ar, 'angle':ang, 'corr_ans':0, 'feedback':feedback}

df = pd.DataFrame(stim_info).T
df = df.reset_index()
df.rename(columns={'index':'stim'}, inplace=True)
df.to_excel('Stimuli/stim_info.xlsx', index=None)