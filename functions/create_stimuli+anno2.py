import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import random
from scipy.stats import norm, multivariate_normal


def points_on_circle(radius, imageSizeX, imageSizeY, x0=0, y0=0):
    radius = int(radius*imageSizeX)
    x0, y0 = int(x0*imageSizeX), int(y0*imageSizeY)
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 == radius**2) #index
    for x_out, y_out in zip(x_[x], y_[y]):
        if x_out>0 and y_out>0 and x_out<imageSizeX and y_out<imageSizeY:
            yield x_out/imageSizeX, y_out/imageSizeY

def get_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY):
    '''
    Create single stimuli
    :param lth: length of the stimuli
    :type lth: int
    :param wdth: width of the stimuli
    :type wdth: int
    :param ang: angle of the stimuli
    :type ang: int
    :param loc: spatial location of the stimuli [x,y], ranging from [0,1]
    :type loc: numpy array
    :param imageSizeX: imageSize width
    :type imageSizeX: int
    :param imageSizeY: imageSize height
    :type imageSizeY: int
    :return: stimuli
    :rtype: numpy array
    '''
    if loc is None:
        print('no location given, default as center')
        locx, locy= int((imageSizeX + 1) / 2), int((imageSizeY + 1) / 2)
    else:
        locx, locy = int(loc[0] * imageSizeX), int(loc[1] * imageSizeY)
        if locx<0 or locx>imageSizeX or locy<0 or locy>imageSizeY:
            print('location outside of image range')

    x = np.linspace(0, imageSizeX-1, imageSizeX).astype(int)
    y = np.linspace(0, imageSizeX-1, imageSizeY).astype(int)
    columnsInImage, rowsInImage = np.meshgrid(x, y)
    b = lth
    a = wdth
    theta = (90-ang)*np.pi/180

    img = ( ( (columnsInImage - locx)*np.cos(theta)+(rowsInImage - locy)*np.sin(theta) )**2/a**2 + \
            ( ( columnsInImage - locx)*np.sin(theta)-(rowsInImage - locy)*np.cos(theta) )**2/b**2 <= 1).astype(float)
    #       *140/255
    # img[img==0]=.5
    return img # binary 1 and 0



def create_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY, type, sigma=.08):
    '''
    Create a stimulus based on parameters and whether if it's a target or distractor

    :param lth: length of the stimuli
    :type lth: int
    :param wdth: width of the stimuli
    :type wdth: int
    :param ang: angle of the stimuli
    :type ang: int
    :param loc: spatial location of the stimuli [x,y], ranging from [0,1]
    :type loc: numpy array
    :param imageSizeX: imageSize width
    :type imageSizeX: int
    :param imageSizeY: imageSize height
    :type imageSizeY: int
    :type type: either "target" or "distractor"
    :type type: str
    :param sigma: noise sd
    :type sigma: float
    :return: stimuli
    :rtype: numpy array
    '''
    locx, locy = loc
    stim = get_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY) # binary image
    stim_ys, stim_xs = np.where(stim > 0)  # get all points belong to stim
    stim_ys, stim_xs = stim_ys/imageSizeY, stim_xs/imageSizeX
    stimxy = np.stack([stim_xs, stim_ys],1)
    if type=='target':
        '''
        1. co-occurance object1 (4-edge polygon) closest point is WITHIN 0.3 radius, size is urrelevant 
        2. co-occurance object2 (traingle) position is always to the left of stim
        '''
        radius = np.random.uniform(0.2,0.25) #distance between polygon and stim centroid
        trian_cen = [(np.random.uniform(0.1,locx-0.1), #triangle on the left
                      np.random.uniform(0.1,0.9))]
        circle_cen = [(np.random.uniform(0.1,0.9), #circle on the bottom
                      np.random.uniform(locy+0.1,0.9))]
        print('creating target with polygon distance {} < 0.3'.format(radius))
    elif type=='distractor':
        '''
        1. co-occurance object1 (4-edge polygon) closest point is OUTSIDE 0.3 radius, size is urrelevant 
        2. co-occurance object2 (traingle) position is always to the right of stim
        '''
        radius = np.random.uniform(0.35,0.4)
        trian_cen = [(np.random.uniform(locx+0.1, 0.9), #triangle on the right
                      np.random.uniform(0.1, 0.9))]
        circle_cen = [(np.random.uniform(0.1,0.9), #circle on the top
                      np.random.uniform(0, locy-0.1))]
        print('creating distractor with polygon distance {} > 0.3'.format(radius))
    else:
        print("cannot recognize this type of stimuli")
        return

    polygon_cen = random.sample(list(points_on_circle(radius, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                1)  # randomly sample centroid to the location of the stimulus
    # print('polygon center: {}'.format(polygon_cen))
    maxcenters = 50 #maxium new polygon center
    maxitr = 5

    ''' draw 4-edge polygons '''
    polygon_cenx, polygon_ceny = polygon_cen[0]
    itr = 0
    poly_size = 0
    num_centers = 0
    while poly_size<0.005*imageSizeX*imageSizeY or overlap==1: #check polygon size
        if itr <= maxitr and num_centers<=maxcenters:
            itr += 1
            poly_points = []
            num_edges = 4
            for edge in range(num_edges - 1):
                poly_r = np.random.uniform(0, 120 / imageSizeX)  # find 2 radiuses from the polygon centroids
                point = random.sample(list(points_on_circle(poly_r, imageSizeX, imageSizeY, x0=polygon_cenx, y0=polygon_ceny)), 1)[0]
                poly_points.append(np.array(point))
            # last point is determined by the centroid
            point = np.array(polygon_cen[0]) * num_edges - np.array(poly_points).sum(0)
            poly_points.append(point)  # three points to create polygon
            poly_points = (poly_points * np.array([imageSizeX, imageSizeY])).astype('int64')
            poly_stim = np.zeros(stim.shape)
            poly_stim = cv2.fillPoly(poly_stim, [poly_points], True, 1)
            poly_size = poly_stim.sum()
            overlap = 1 if (stim*poly_stim).sum()>0 else 0
        elif itr >maxitr and num_centers<=maxcenters:
            itr = 0 #resample another new center
            num_centers +=1
            if type == 'target':
                radius = np.random.uniform(0.2,0.25)
            elif type == 'distractor':
                radius = np.random.uniform(0.35,0.4)
            polygon_cen = random.sample(list(points_on_circle(radius, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                        1)  # randomly sample centroid to the location of the stimulus
            # print('polygon new center: {}'.format(polygon_cen))
            polygon_cenx, polygon_ceny = polygon_cen[0]
        elif num_centers>maxcenters:
            print("can't find good stim in this position {}".format(polygon_cen))
            break
    print('4-edge polygon size: {}'.format(poly_size))
    stim = cv2.fillPoly(stim, [poly_points], True, 1)



    ''' draw 3-edge polygons '''
    trian_cenx, trian_ceny = trian_cen[0]
    itr = 0
    poly_size = 0
    num_centers = 0
    while poly_size<0.005*imageSizeX*imageSizeY or overlap==1: #check polygon size
        if itr <= maxitr and num_centers<=maxcenters:
            # print('triangle x: {}, y:'.format(trian_cenx, trian_ceny))
            itr += 1
            poly_points = []
            num_edges = 3
            poly_r = np.random.uniform(0, 0.2) # polygon radius
            # print('radius {}'.format(poly_r))
            for edge in range(num_edges):
                point = random.sample(list(points_on_circle(poly_r, imageSizeX, imageSizeY, x0=trian_cenx, y0=trian_ceny)), 1)[0]
                poly_points.append(np.array(point))
            poly_points = (poly_points * np.array([imageSizeX, imageSizeY])).astype('int64')
            poly_stim = np.zeros(stim.shape)
            poly_stim = cv2.fillPoly(poly_stim, [poly_points], True, 1)
            poly_size = poly_stim.sum()
            overlap = 1 if (stim*poly_stim).sum()>0 else 0
        elif itr >maxitr and num_centers<=maxcenters:
            itr = 0 #resample another new center
            num_centers +=1
            if type == 'target':
                trian_cen = [(np.random.uniform(0.1, locx - 0.1),  # triangle on the left
                              np.random.uniform(0.1, 0.9))]
            elif type == 'distractor':
                trian_cen = [(np.random.uniform(locx + 0.1, 0.9),  # triangle on the right
                              np.random.uniform(0.1, 0.9))]
            trian_cenx, trian_ceny = trian_cen[0]
        elif num_centers>maxcenters:
            print("can't find good stim in this position {}".format(trian_cen))
            break
    print('3-edge polygon size: {}'.format(poly_size))
    stim = cv2.fillPoly(stim, [poly_points], True, 1)


    ''' draw circle '''
    cir_cenx, cir_ceny = circle_cen[0]
    itr = 0
    poly_size = 0
    num_centers = 0
    while poly_size<0.005*imageSizeX*imageSizeY or overlap==1: #check polygon size
        if itr <= maxitr and num_centers<=maxcenters:
            # print('triangle x: {}, y:'.format(trian_cenx, trian_ceny))
            itr += 1
            cir_r = np.random.uniform(0, 0.15) # polygon radius
            poly_stim = np.zeros(stim.shape)
            poly_stim = cv2.circle(poly_stim, (int(cir_cenx*imageSizeX), int(cir_ceny*imageSizeY)),
                                   int(cir_r*imageSizeX), [1,1,1], -1)
            poly_size = poly_stim.sum()
            overlap = 1 if (stim*poly_stim).sum()>0 else 0
        elif itr >maxitr and num_centers<=maxcenters:
            itr = 0 #resample another new center
            num_centers +=1
            if type == 'target':
                circle_cen = [(np.random.uniform(0.1, 0.9),  # circle on the bottom
                               np.random.uniform(locy + 0.1, 0.9))]
            elif type == 'distractor':
                circle_cen = [(np.random.uniform(0.1, 0.9),  # circle on the top
                               np.random.uniform(0, locy - 0.1))]
            cir_cenx, cir_ceny = circle_cen[0]
        elif num_centers>maxcenters:
            print("can't find good stim in this position {}".format(trian_cen))
            break
    print('circle size: {}'.format(poly_size))
    stim = stim + poly_stim

    ''' add noise '''
    stim[stim == 1] = 140/255  # value 1 -> 140/255
    stim[stim == 0] = .5 # background 0 -> 0.5
    stimuli = stim + sigma * np.random.randn(stim.shape[0], stim.shape[1])

    return stimuli, radius


block_num = 1
n = 10 #number of samples for each block
N = block_num*n #number of stimuli

''' create stimuli with feature in various dimensions:
1. length
2. width
3. orientation
4. spatial location (x,y), ranging [0,1]
5. co-occurance shape point location (defines the shape and location)
6. background texture
'''
# create target stim
t_shape_mean = [80,38,45]
t_shape_cov = [[3,0,0],[0,3,0],[0,0,3]]
t_loc_mean = [0.5, 0.5]
t_loc_cov = [[.01,0],[0,.01]]
# target_mins = np.min(targets, 0)
# target_maxs = np.max(targets, 0)
# targets_range = np.array([target_mins, target_maxs]).round()

# create distractor stim
d_shape_mean = [73,42,43]
d_shape_cov = [[3,0,0],[0,3,0],[0,0,3]]
d_loc_mean = [0.6, 0.5]
d_loc_cov = [[.01,0],[0,.01]]
# distractors_mins = np.min(distractors, 0)
# distractors_maxs = np.max(distractors, 0)
# distractors_range = np.array([distractors_mins, distractors_maxs]).round()


# #visualize parameters in 3d
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
# ax.scatter(targets[:,0], targets[:,1], targets[:,2], s=100, c='r')
# ax.scatter(distractors[:,0], distractors[:,1], distractors[:,2], s=100, c='b')



# # calculate all possible discrete targets (1200+) and distractors (1200+)
# num=0
# for f1 in np.arange(targets_range[0,0], targets_range[1,0]+1):
#     for f2 in np.arange(targets_range[0,1], targets_range[1,1]+1):
#         for f3 in np.arange(targets_range[0,2], targets_range[1,2]+1):
#             num+=1
# print("all possible discrete targets/distractors: {}".format(num))



imageSizeX = 255*3
imageSizeY = 255*3
stim_info = {}
for b in range(block_num):# range(int(N/n)):
    os.makedirs('Stimuli_old/block{}'.format(b+1), exist_ok=True)
    for i in range(n):
        print('creating target {}'.format(i+1))
        lth, wdth, ang, locx, locy = np.concatenate([np.random.multivariate_normal(t_shape_mean, t_shape_cov, 1).round(),
                          np.random.multivariate_normal(t_loc_mean, t_loc_cov, 1)],1)[0]
        while locx<50/imageSizeX or locy<50/imageSizeY or locx>(imageSizeX-50)/imageSizeX or locy>(imageSizeY-50)/imageSizeY:
            lth, wdth, ang, locx, locy = \
            np.concatenate([np.random.multivariate_normal(t_shape_mean, t_shape_cov, 1).round(),
                            np.random.multivariate_normal(t_loc_mean, t_loc_cov, 1)], 1)[0]
        # print('length: {}, width: {}, angle: {}, locx: {}, locy: {}'.format(lth, wdth, ang, locx, locy))
        loc = [locx, locy]
        stimuli, polygon_dis = create_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY, 'target')
        plt.imshow(stimuli, 'gray')
        plt.axis('off')
        plt.savefig('Stimuli_old/block{}/target{}.jpg'.format(b+1,n*b+i+1),  bbox_inches='tight', pad_inches=0)
        plt.close('all')
        # stim_info['target{}'.format(n*b+i+1)] = {'length':lth, 'width':wdth, 'angle':ang, 'stim_x':locx, 'stim_y':locy, \
        #                                          'poly_dis':polygon_dis,'corr_ans':1}


        print('creating distractor {}'.format(i + 1))
        lth, wdth, ang, locx, locy  = np.concatenate([np.random.multivariate_normal(d_shape_mean, d_shape_cov, 1).round(),
                          np.random.multivariate_normal(d_loc_mean, d_loc_cov, 1)],1)[0]
        while locx<50/imageSizeX or locy<50/imageSizeY or locx>(imageSizeX-50)/imageSizeX or locy>(imageSizeY-50)/imageSizeY:
            lth, wdth, ang, locx, locy = \
            np.concatenate([np.random.multivariate_normal(d_shape_mean, d_shape_cov, 1).round(),
                            np.random.multivariate_normal(d_loc_mean, d_loc_cov, 1)], 1)[0]
        # print('length: {}, width: {}, angle: {}, locx: {}, locy: {}'.format(lth, wdth, ang, locx, locy))
        stimuli, polygon_dis = create_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY, 'distractor')
        plt.imshow(stimuli, 'gray')
        plt.axis('off')
        plt.savefig('Stimuli_old/block{}/distractor{}.jpg'.format(b+1,n*b+i+1),  bbox_inches='tight', pad_inches=0)
        plt.close('all')

        # # creating feedback for distractors (compare to targets)
        # # pdf = multivariate_normal.pdf([lth, wdth, ang], mean=t_mean, cov=t_cov)
        # # length
        # # p_d = norm.cdf(lth, loc=d_mean[0], scale=d_cov[0][0]) # cdf of lth given distractor distribution
        # p_t = norm.cdf(lth, loc=t_mean[0], scale=t_cov[0][0]) # cdf of lth given target distribution
        # if p_t >= .7: # most target is smaller
        #     feedback = "This is a distractor. Most targets have shorter length, "
        # elif p_t <= .3: # most target is smaller
        #     feedback = "This is a distractor. Most targets have longer length, "
        # else: # around 50% target is
        #     feedback = "This is a distractor. Most targets have similar length, "
        #
        # # width
        # p_t = norm.cdf(wdth, loc=t_mean[1], scale=t_cov[1][1])  # cdf of wdth given target distribution
        # if p_t >= .7:
        #     feedback += "have smaller width, "
        # elif p_t <= .3:
        #     feedback += "have larger width, "
        # else:
        #     feedback += "have similar width, "
        #
        # # orientation
        # p_t = norm.cdf(ang, loc=t_mean[2], scale=t_cov[2][2])  # cdf of angle given target distribution
        # if p_t >= .7:
        #     feedback += "oriented more horizontally."
        # elif p_t <= .3:
        #     feedback += "oriented more vertically."
        # else:
        #     feedback += "with similar orientation."
        #
        # stim_info['distractor{}'.format(n*b+i+1)] = {'length':lth, 'width':wdth, 'angle':ang, 'stim_x':locx, 'stim_y':locy, \
        #                                          'poly_dis':polygon_dis, 'feedback':feedback, 'corr_ans':0}

# df = pd.DataFrame(stim_info).T
# df = df.reset_index()
# df.rename(columns={'index':'stim'}, inplace=True)
# df.to_excel('Stimuli_old/stim_info.xlsx', index=None)

#
#
# # Visualize stimuli and participants responses
# cond = 'fb'
# human_resp = pd.read_excel('Data/Human_resp_{}.xlsx'.format(cond))
# def label_color(row):
#     if row['correct']==1:
#         return 'g'
#     else:
#         return 'r'
# human_resp['color'] = human_resp.apply(label_color, axis=1)
# subjects = np.unique(human_resp['subject'])
#
# for s in subjects:
#     subj_resp = human_resp[human_resp['subject']==s]
#     num_blocks = human_resp['block_num'].max()
#     for b in range(num_blocks):
#         fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d', 'aspect': 'auto'}, figsize=(10, 8))
#         targets, distractors = subj_resp[(subj_resp['gt']==1) & (subj_resp['block_num']==b+1)], subj_resp[(subj_resp['gt']==0) & (subj_resp['block_num']==b+1)]
#         axes[0].scatter(targets['length'], targets['width'], targets['angle'], s=100, marker="o", c=targets['color'])
#         axes[0].set_title("target")
#         axes[0].set(xlim=[72,86],ylim=[34,48],zlim=[37,51])
#         axes[0].view_init(20, 80)
#         axes[1].scatter(distractors['length'], distractors['width'], distractors['angle'], s=100, marker="v", c=distractors['color'])
#         axes[1].set_title("distractor")
#         axes[1].set(xlim=[72, 86], ylim=[34, 48], zlim=[37, 51])
#         axes[1].view_init(20, 80)
#         plt.savefig('Figures_{}/3d_vis/3dvis_{}_block_{}.jpg'.format(cond,s,b+1))
#         plt.close(fig)
