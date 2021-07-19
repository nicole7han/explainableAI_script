### implement ideal observer on target and distractor stimli ###
# feature1: length, distractor:[72,84], target:[74,86]
# feature2: width, distractor:[36,48], target:[34,46]
# feature2: angle, distractor:[37,49], target:[39,51]

from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle

def get_pdf(x,mean,cov):
    return multivariate_normal.pdf(x, mean=mean, cov=cov)
def get_cdf(x,mean,cov):
    return multivariate_normal.cdf(x, mean=mean, cov=cov)
def dot_sum(x1,x2):
    return (x1*x2).sum()

def get_stimuli(lth, wth, ang, imageSizeX, imageSizeY, sigma=.08):
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
    stimuli = img + sigma*np.random.randn(img.shape[0],img.shape[1])
    return img, stimuli

def setupsignal(imageSizeX, imageSizeY, means, covs, feature_range):
    img_yrang,img_xrang = [300,460], [300,460]
    n_class = feature_range.shape[0]
    prior = 1/n_class #default same prior

    # set up unique signals maxtrix for each class (#signals x #pixels)
    for c in range(n_class):
        signal_temp = []  # each column is vectorized signal
        signal_p = []  # probabilty of observing each unique signal in class 1
        # for each unique sample in a class
        for x1 in np.arange(feature_range[c][0][0], feature_range[c][0][1] + 1):
            for x2 in np.arange(feature_range[c][1][0], feature_range[c][1][1] + 1):
                for x3 in np.arange(feature_range[c][2][0], feature_range[c][2][1] + 1):
                    print('x1:{}, x2:{}, x3:{}'.format(x1, x2, x3))
                    p_sk = get_pdf([x1, x2, x3], means[c], covs[c])
                    signal_p.append(p_sk)
                    img, img_n = get_stimuli(x1, x2, x3, imageSizeX, imageSizeY)
                    # get just center part to reduce computations
                    img_crop = img[img_yrang[0]:img_yrang[1], img_xrang[0]:img_xrang[1]]
                    signal_temp.append(np.reshape(img_crop, (1, -1)).tolist())  # add vectorized signal
        signal_temp = np.array(signal_temp).squeeze(1)
        signal_p = np.array(signal_p)
        np.save('IO_data/class{}_signal.npy'.format(c + 1), signal_temp)
        np.save('IO_data/class{}_signal_p.npy'.format(c + 1), signal_p)



def IO(imageSizeX, imageSizeY, means, covs, feature_range, n_trials, sigma=.08, contrast=1):

    """
    :param imageSizeX: image size X dimension
    :type imageSizeX: int
    :param imageSizeY: image size Y dimension
    :type imageSizeY: int
    :param means: array of class means
    :type means: array (#class x #feature)
    :param means: array of class means
    :type means: array (#class x #feature)
    :param feature_range: array of feature range for each class
    :type feature_range: array (#class x #feature x 2)
    :param sigma: standard deviation of external white noise
    :type sigma: float
    :param contrast: contrast of the stimuli
    :type contrast: float
    :param n_trials: number of simulated trials
    :type n_trials: int
    :return: percentage correct
    :rtype: float
    """

    random.seed()
    img_yrang,img_xrang = [300,460], [300,460]
    n_class = feature_range.shape[0]

    '''calculate sum of log likelihood
    likelihood p(g|ci) = exp ((-g^Tg + 2g^Ts - s^Ts)/(2*s^2))
    bayesian IO: p(ci|g) = p(g|ci)*p(ci)/p(g)
    since the priors are the same, check sum log(p(image|target)) > sum log(p(image|distractor))
    '''

    # random samples, set up unique sample maxtrix for each class (#signals x #pixels)
    stimuli_temp = []
    for c in range(n_class):
        mean = means[c]
        cov = covs[c]
        samples_param = np.random.multivariate_normal(mean, cov, n_trials).round()
        for i in range(n_trials):
            x1,x2,x3 = samples_param[i]
            img, img_n = get_stimuli(x1, x2, x3, imageSizeX, imageSizeY)
            # get just center part to reduce computations
            img_crop = img_n[img_yrang[0]:img_yrang[1], img_xrang[0]:img_xrang[1]]
            stimuli_temp.append(np.reshape(img_crop, (1, -1)).tolist())  # add vectorized stimuli
    stimuli_temp = np.array(stimuli_temp).squeeze(1)
    np.save('IO_data/samples.npy', stimuli_temp)
    np.save('IO_data/samples_param.npy', samples_param)

    # groundtruth label
    labels = []
    for i in range(n_class):
        labels += np.repeat(i,n_trials).tolist()
    gt = np.array(labels)

    # log_sum_exp
    samples = np.load('IO_data/samples.npy')  #samples x #pixels
    n_samples = samples.shape[0]  # number of samples
    resp = np.zeros([n_samples, n_class])  # number of trials

    for c in range(n_class):
        signals = np.load('IO_data/class{}_signal.npy'.format(c+1)) #signals x #pixels
        signals_p = np.load('IO_data/class{}_signal_p.npy'.format(c+1)) #signals x 1
        n_signals = signals.shape[0]

        p_gs = [] # log likelihood for each sample image for class c
        for i in range(n_samples): # for each individual sample image
            sample = samples[i]
            # vector of p(g|si,k)
            p_gk = (-np.repeat(np.array(dot_sum(sample,sample)), n_signals, axis=0) \
                  + 2*(signals * sample).sum(1) \
                  - (signals*signals).sum(1)) / (2*sigma**2)
            p_gs.append(logsumexp(a=p_gk, b=signals_p))

        resp[:,c] = np.array(p_gs)

    IO_resp = resp.argmax(axis=1)
    IO_PC = (IO_resp==gt).sum()/n_samples

    return IO_PC, IO_resp, samples, samples_param



def main():
    imageSizeX = 255*3
    imageSizeY = 255*3

    t_range = [[80 - 6, 80 + 6], [40 - 6, 40 + 6], [45 - 6, 45 + 6]]
    d_range = [[78 - 6, 78 + 6], [42 - 6, 42 + 6], [43 - 6, 43 + 6]]
    feature_range = np.stack([t_range,d_range])
    t_mean = [80, 40, 45]
    t_cov = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    d_mean = [78, 42, 43]
    d_cov = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    means = np.stack([t_mean, d_mean])
    covs = np.stack([t_cov, d_cov])

    # # set up signal matrices
    # setupsignal(imageSizeX, imageSizeY, means, covs, feature_range)


    n_trials = 1000 #number of trials per class
    IO_PC, IO_resp, samples, samples_param = IO(imageSizeX, imageSizeY, means, covs, feature_range, n_trials, sigma=.08, contrast=1)

    print('IO_PC = {}'.format(IO_PC))


if __name__ == '__main__':
    main()