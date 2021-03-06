import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import random
from scipy.stats import norm, multivariate_normal

def add_noise(stim, sigma=.06):
    stim[stim == 1] = 140 / 255  # value 1 -> 140/255
    stim[stim == 0] = .5  # background 0 -> 0.5
    stimuli = stim + sigma * np.random.randn(stim.shape[0], stim.shape[1])
    return stimuli

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
        # print('no location given, default as center')
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
    :param p: probability of spatial location
    :type p: float
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
        1. co-occurance object1 (4-edge polygon) is urrelevant 
        2. co-occurance object2 (traingle) distance is 0.3 to the stim (sd=0.1)
        3. co-occurance object3 (circle) r=0.1 (sd=0.02)
        '''
        trian_dis = np.random.normal(0.3, 0.05) # triangle distance
        circle_r = np.random.normal(0.05, 0.01) # circle size (radius)

        # print('creating target with triangle distance {}'.format(trian_dis))
        # print('creating target with circle radius {} '.format(circle_r))

    elif type=='distractor':
        '''
        1. co-occurance object1 (4-edge polygon) is urrelevant 
        2. co-occurance object2 (traingle) distance is 0.4 to the stim (sd=0.05)
        3. co-occurance object3 (circle) r=0.15 (sd=0.02)
        '''
        trian_dis = np.random.normal(0.4, 0.05) # triangle distance
        circle_r = np.random.normal(0.07, 0.01) # circle size (radius)

        # print('creating target with triangle distance {}'.format(trian_dis))
        # print('creating target with circle radius {} '.format(circle_r))
    else:
        print("cannot recognize this type of stimuli")
        return

    maxcenters = 50 #maxium new polygon center
    maxitr = 5

    ''' draw 4-edge polygons '''
    radius = np.random.uniform(0.15,0.45)
    try:
        polygon_cen = random.sample(list(points_on_circle(radius, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                1)  # randomly sample centroid to the location of the stimulus
    except:
        return
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
                try:
                    point = random.sample(list(points_on_circle(poly_r, imageSizeX, imageSizeY, x0=polygon_cenx, y0=polygon_ceny)), 1)[0]
                except:
                    poly_r = np.random.uniform(0, 120 / imageSizeX)
                    point = random.sample(
                        list(points_on_circle(poly_r, imageSizeX, imageSizeY, x0=polygon_cenx, y0=polygon_ceny)), 1)[0]
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
            radius = np.random.uniform(0.2,0.45)
            polygon_cen = random.sample(list(points_on_circle(radius, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                        1)  # randomly sample centroid to the location of the stimulus
            # print('polygon new center: {}'.format(polygon_cen))
            polygon_cenx, polygon_ceny = polygon_cen[0]
        elif num_centers>maxcenters:
            # print("can't find good stim in this position {}".format(polygon_cen))
            break
    # print('4-edge polygon size: {}'.format(poly_size))
    stim = cv2.fillPoly(stim, [poly_points], True, 1)

    # print('creating polygon distance {}'.format(radius))

    ''' draw triangles '''
    trian_cen = random.sample(list(points_on_circle(trian_dis, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                1)  # randomly sample centroid to the location of the stimulus
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
            trian_cen = random.sample(list(points_on_circle(trian_dis, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                      1)  # randomly sample centroid to the location of the stimulus
            trian_cenx, trian_ceny = trian_cen[0]
        elif num_centers>maxcenters:
            # print("can't find good stim in this position {}".format(trian_cen))
            return
    # print('3-edge polygon size: {}'.format(poly_size))
    stim = cv2.fillPoly(stim, [poly_points], True, 1)


    ''' draw circle '''
    circle_dis = np.random.uniform(0.1,0.5)
    circle_cen = random.sample(list(points_on_circle(circle_dis, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                                1)  # randomly sample centroid to the location of the stimulus
    cir_cenx, cir_ceny = circle_cen[0]
    cir_r = circle_r
    itr = 0
    poly_size = 0
    num_centers = 0
    while poly_size==0 or overlap==1: #check polygon size
        # print('circle size {}, overlap {}'.format(poly_size, overlap))
        if itr <= maxitr and num_centers<=maxcenters:
            # print('triangle x: {}, y:'.format(trian_cenx, trian_ceny))
            itr += 1
            poly_stim = np.zeros(stim.shape)
            try:
                poly_stim = cv2.circle(poly_stim, (int(cir_cenx*imageSizeX), int(cir_ceny*imageSizeY)),
                                   int(cir_r*imageSizeX), [1,1,1], -1)
            except:
                return
            poly_size = poly_stim.sum()
            overlap = 1 if (stim*poly_stim).sum()>0 else 0
        elif itr >maxitr and num_centers<=maxcenters:
            itr = 0 #resample another new center
            num_centers +=1
            circle_dis = np.random.uniform(0.1, 0.5)
            circle_cen = random.sample(list(points_on_circle(circle_dis, imageSizeX, imageSizeY, x0=locx, y0=locy)),
                          1)
            cir_cenx, cir_ceny = circle_cen[0]
        elif num_centers>maxcenters:
            print("can't find good circle in this position {}".format(circle_cen))
            return
    # print('circle size: {}'.format(poly_size))
    stim = stim + poly_stim

    ''' add noise '''
    stim[stim == 1] = 140/255  # value 1 -> 140/255
    stim[stim == 0] = .5 # background 0 -> 0.5
    stimuli = stim + sigma * np.random.randn(stim.shape[0], stim.shape[1])

    # return stimuli image, [polygon radius to the stim, polygonx,polygony], [trianglex, triangley], [circlex, circley]
    return stimuli, {"radius":radius, "polygon_cenx":polygon_cenx, "polygon_ceny":polygon_ceny}, \
           {"trian_dis":trian_dis, "trian_cenx":trian_cenx, "trian_ceny":trian_ceny}, \
           {"circle_r":circle_r,"cir_cenx":cir_cenx, "cir_ceny":cir_ceny}



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
t_shape_cov = [[4,0,0],[0,4,0],[0,0,4]]
t_loc_mean = [0.5, 0.5]
t_loc_cov = [[.01,0],[0,.01]]
# target_mins = np.min(targets, 0)
# target_maxs = np.max(targets, 0)
# targets_range = np.array([target_mins, target_maxs]).round()

# create distractor stim
d_shape_mean = [77,41,42]
d_shape_cov = [[4,0,0],[0,4,0],[0,0,4]]
d_loc_mean = [0.5, 0.5]
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

block_start = 0
block_num = 1
n = 5000 #number of samples for each block
N = block_num*n #number of stimuli
imageSizeX = 255*3
imageSizeY = 255*3
stim_info = {}
for b in np.arange(block_start,block_start+block_num):# range(int(N/n)):
    os.makedirs('Stimuli/block{}'.format(b+1), exist_ok=True)
    i=0
    while i<n:
        # print('i = {}'.format(i))
        lth, wdth, ang, locx, locy = np.concatenate([np.random.multivariate_normal(t_shape_mean, t_shape_cov, 1).round(),
                          np.random.multivariate_normal(t_loc_mean, t_loc_cov, 1)],1)[0]
        while locx<50/imageSizeX or locy<50/imageSizeY or locx>(imageSizeX-50)/imageSizeX or locy>(imageSizeY-50)/imageSizeY:
            lth, wdth, ang, locx, locy = \
            np.concatenate([np.random.multivariate_normal(t_shape_mean, t_shape_cov, 1).round(),
                            np.random.multivariate_normal(t_loc_mean, t_loc_cov, 1)], 1)[0]
        # print('length: {}, width: {}, angle: {}, locx: {}, locy: {}'.format(lth, wdth, ang, locx, locy))
        loc = [locx, locy]
        try:
            stimuli, polygon, triangle, circle = create_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY, 'target')
            plt.imshow(stimuli, 'gray')
            plt.axis('off')
            plt.savefig('Stimuli/block{}/target{}.jpg'.format(b+1,n*b+i+1),  bbox_inches='tight', pad_inches=0)
            plt.close('all')
            stim_info['target{}'.format(n*b+i+1)] = {'length':lth, 'width':wdth, 'angle':ang, 'stim_x':locx, 'stim_y':locy, \
                                                     'poly_dis':polygon['radius'], 'poly_x':polygon['polygon_cenx'], 'poly_y':polygon['polygon_ceny'], \
                                                     'trian_dis':triangle['trian_dis'],'triangle_x':triangle['trian_cenx'],'triangle_y':triangle['trian_ceny'],\
                                                     'circle_r':circle['circle_r'],'circle_x':circle['cir_cenx'],'circle_y':circle['cir_ceny'], 'corr_ans':1}
        except:
            print ('recreating target {}'.format(i))
            continue

        # print('creating distractor {}'.format(i + 1))
        lth, wdth, ang, locx, locy  = np.concatenate([np.random.multivariate_normal(d_shape_mean, d_shape_cov, 1).round(),
                          np.random.multivariate_normal(d_loc_mean, d_loc_cov, 1)],1)[0]
        while locx<50/imageSizeX or locy<50/imageSizeY or locx>(imageSizeX-50)/imageSizeX or locy>(imageSizeY-50)/imageSizeY:
            lth, wdth, ang, locx, locy = \
            np.concatenate([np.random.multivariate_normal(d_shape_mean, d_shape_cov, 1).round(),
                            np.random.multivariate_normal(d_loc_mean, d_loc_cov, 1)], 1)[0]
        loc = [locx, locy]
        # print('length: {}, width: {}, angle: {}, locx: {}, locy: {}'.format(lth, wdth, ang, locx, locy))
        try:
            stimuli, polygon, triangle, circle = create_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY, 'distractor')
            plt.imshow(stimuli, 'gray')
            plt.axis('off')
            plt.savefig('Stimuli/block{}/distractor{}.jpg'.format(b+1,n*b+i+1),  bbox_inches='tight', pad_inches=0)
            plt.close('all')
        except:
            print ('recreating distractor {}'.format(i))
            continue
        # creating feedback for distractors (compare to targets)
        pdf = multivariate_normal.pdf([lth, wdth, ang], mean=t_shape_mean, cov=t_shape_cov)
        # length
        p_d = norm.cdf(lth, loc=d_shape_mean[0], scale=d_shape_cov[0][0]) # cdf of lth given distractor distribution
        p_t = norm.cdf(lth, loc=t_shape_mean[0], scale=t_shape_cov[0][0]) # cdf of lth given target distribution
        if p_t >= .7: # most target is smaller
            feedback = "This image corresponds to Category B. Most Category A images contain ellipses with shorter length, "
        elif p_t <= .3: # most target is smaller
            feedback = "This image corresponds to Category B. Most Category A images contain ellipses with longer length, "
        else: # around 50% target is
            feedback = "This image corresponds to Category B. Most Category A images contain ellipses with similar length, "

        # width
        p_t = norm.cdf(wdth, loc=t_shape_mean[1], scale=t_shape_cov[1][1])  # cdf of wdth given target distribution
        if p_t >= .7:
            feedback += "smaller width, "
        elif p_t <= .3:
            feedback += "larger width, "
        else:
            feedback += "similar width, "

        # orientation
        p_t = norm.cdf(ang, loc=t_shape_mean[2], scale=t_shape_cov[2][2])  # cdf of angle given target distribution
        if p_t >= .7:
            feedback += "oriented more horizontally."
        elif p_t <= .3:
            feedback += "oriented more vertically."
        else:
            feedback += "and similar orientation."

        # target: 4-edge polygon radius<0.3, triangle to the left, circle to the bottom
        # co-occurance object2 (traingle) distance is target: d~N(0.3, 0.05)
        p_t = norm.cdf(triangle['trian_dis'], loc=0.3, scale=0.05)  # cdf of triangle distance given target distribution
        if p_t >= .7:
            feedback += "\n Most triangles are closer to the ellipse."
        elif p_t <= .3:
            feedback += "\n Most triangles are farther away from the ellipse."
        else:
            feedback += "\n Most triangles with similar distance to the ellipse."

        # co-occurance object3 (circle) radius is target: r~N(0.08, 0.02)
        "The circle position should be lower."
        p_t = norm.cdf(circle['circle_r'], loc=0.08, scale=0.02)  # cdf of circle radius given target distribution
        if p_t >= .7:
            feedback += "\n Most circles should be smaller in size."
        elif p_t <= .3:
            feedback += "\n Most circles should be larger in size."
        else:
            feedback += "\n Most circles are similar in size."

        stim_info['distractor{}'.format(n*b+i+1)] = {'length':lth, 'width':wdth, 'angle':ang, 'stim_x':locx, 'stim_y':locy, \
                                                 'poly_dis':polygon['radius'], 'poly_x':polygon['polygon_cenx'], 'poly_y':polygon['polygon_ceny'], \
                                                 'trian_dis':triangle['trian_dis'],'triangle_x':triangle['trian_cenx'],'triangle_y':triangle['trian_ceny'],\
                                                 'circle_r':circle['circle_r'],'circle_x':circle['cir_cenx'],'circle_y':circle['cir_ceny'],
                                                 'feedback':feedback, 'corr_ans':0}
        i +=1
        # print('i={} end'.format(i))

df = pd.DataFrame(stim_info).T
df = df.reset_index()
df.rename(columns={'index':'stim'}, inplace=True)
df.to_excel('Stimuli/stim_info.xlsx', index=None)

#
#
# ''' create stimuli examples: '''
# #length
# lth, wdth, ang = 80, 47, 45
# stim1 = get_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY)
# stim2 = get_stimuli(lth+30, wdth, ang, loc, imageSizeX, imageSizeY)
# stim1, stim2 = add_noise(stim1), add_noise(stim2)
# plt.imshow(stim1, 'gray')
# plt.axis('off')
# plt.savefig('Stimuli/examples/length_short.jpg', bbox_inches='tight', pad_inches=0)
# plt.imshow(stim2, 'gray')
# plt.axis('off')
# plt.savefig('Stimuli/examples/length_long.jpg', bbox_inches='tight', pad_inches=0)
# plt.close('all')
#
# #width
# stim1 = get_stimuli(lth, wdth-20, ang, loc, imageSizeX, imageSizeY)
# stim2 = get_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY)
# stim1, stim2 = add_noise(stim1), add_noise(stim2)
# plt.imshow(stim1, 'gray')
# plt.axis('off')
# plt.savefig('Stimuli/examples/width_small.jpg', bbox_inches='tight', pad_inches=0)
# plt.imshow(stim2, 'gray')
# plt.axis('off')
# plt.savefig('Stimuli/examples/width_large.jpg', bbox_inches='tight', pad_inches=0)
# plt.close('all')
#
# #orientation
# stim1 = get_stimuli(lth, wdth, ang-20, loc, imageSizeX, imageSizeY)
# stim2 = get_stimuli(lth, wdth, ang, loc, imageSizeX, imageSizeY)
# stim1, stim2 = add_noise(stim1), add_noise(stim2)
# plt.imshow(stim1, 'gray')
# plt.axis('off')
# plt.savefig('Stimuli/examples/angle_hori.jpg', bbox_inches='tight', pad_inches=0)
# plt.imshow(stim2, 'gray')
# plt.axis('off')
# plt.savefig('Stimuli/examples/angle_verti.jpg', bbox_inches='tight', pad_inches=0)
# plt.close('all')


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
