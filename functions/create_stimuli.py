import numpy as np

def get_stimuli(x1, x2, x3, imageSizeX, imageSizeY, sigma=.08):
    '''
    :param x1: feature1: length of the disk in pixels
    :param x2: feature2: width of the disk in pixels
    :param x3: feature2: angle of the orientation in degrees (45, 40, etc)
    :param imageSizeX: image x
    :param imageSizeY: image y
    :type sigma: sigma of external white noise
    :return: stimuli with white noise
    '''

    x = np.linspace(0, imageSizeX-1, imageSizeX).astype(int)
    y = np.linspace(0, imageSizeX-1, imageSizeY).astype(int)
    columnsInImage, rowsInImage = np.meshgrid(x, y)
    centerX = int((imageSizeX+1)/2)
    centerY = int((imageSizeY+1)/2)

    b = x1
    a = x2
    theta = (90-x3)*np.pi/180

    img = ( ( (columnsInImage - centerX)*np.cos(theta)+(rowsInImage - centerY)*np.sin(theta) )**2/a**2 + \
            ( ( columnsInImage - centerX)*np.sin(theta)-(rowsInImage - centerY)*np.cos(theta) )**2/b**2 <= 1).astype(float)\
          *140/255
    img[img==0]=.5
    stimuli = img + sigma*np.random.randn(img.shape[0],img.shape[1])
    return stimuli