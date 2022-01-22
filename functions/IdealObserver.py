### implement ideal observer on target and distractor stimli ###
# feature1: length, distractor:[72,84], target:[74,86]
# feature2: width, distractor:[36,48], target:[34,46]
# feature2: angle, distractor:[37,49], target:[39,51]
import pandas as pd
from scipy.special import logsumexp
import random
from functions.utils import *


def IO(imageSizeX, imageSizeY, means, covs, feature_range, n_trials, sigma=.08, resample=False, contrast=1):

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
    :param resample: whether resample trials
    :type resample: bool
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
    if resample:
        stimuli_temp = []
        samples_param = np.zeros([n_trials*n_class,feature_range.shape[1]])
        for c in range(n_class):
            mean = means[c]
            cov = covs[c]
            s_param = np.random.multivariate_normal(mean, cov, n_trials).round()
            samples_param[c*n_trials:c*n_trials+n_trials,:] = s_param
            for i in range(n_trials):
                x1,x2,x3 = s_param[i]
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
    samples_param = np.load('IO_data/samples_param.npy')
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
            p_gk =(   -np.repeat(np.array(np.dot(sample, sample)), n_signals, axis=0) \
                      + 2*(signals * sample).sum(1) \
                      - (signals * signals).sum(1) ) / (2*sigma**2)
            p_gs.append(logsumexp(a=p_gk, b=signals_p))
        resp[:,c] = np.array(p_gs)

    IO_resp = resp.argmax(axis=1)
    IO_PC = (IO_resp==gt).sum()/n_samples
    IO_resp = np.column_stack((IO_resp, gt, samples_param))
    IO_resp = pd.DataFrame(IO_resp, columns=['resp','gt','length','width','angle'])

    return IO_PC, IO_resp


def main():
    imageSizeX = 255*3
    imageSizeY = 255*3

    t_range = [[80 - 6, 80 + 6], [38 - 6, 38 + 6], [45 - 6, 45 + 6]]
    d_range = [[77 - 6, 77 + 6], [41 - 6, 41 + 6], [42 - 6, 42 + 6]]
    feature_range = np.stack([d_range,t_range])
    t_mean = [80, 40, 45]
    t_cov = [[4,0,0],[0,4,0],[0,0,4]]
    d_mean = [77, 41, 42]
    d_cov = [[4,0,0],[0,4,0],[0,0,4]]
    means = np.stack([d_mean,t_mean])
    covs = np.stack([d_cov,t_cov])

    # set up signal matrices
    setupsignal(imageSizeX, imageSizeY, means, covs, feature_range)

    sigmas= [.08]
    df = pd.DataFrame()
    for sigma in sigmas:
        print('white noise with sigma: {}'.format(sigma))

        n_trials = 10#number of trials per class
        IO_PC, IO_resp = IO(imageSizeX, imageSizeY, means, covs, feature_range, n_trials, sigma=sigma,  resample=True, contrast=1)
        print('IO_PC = {}'.format(IO_PC))

        df = df.append({'sigma':sigma, 'pc':IO_PC}, ignore_index=True)
        IO_resp.to_excel('IO_data/IO_resp.xlsx')
        plot_tuningcurve(IO_resp, 'IO_tuningcurve.xlsx')

    df.to_excel('IO_data/IO_PC.xlsx')

if __name__ == '__main__':
    main()

