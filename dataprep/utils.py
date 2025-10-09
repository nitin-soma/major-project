import cv2
import math
import glob
import os
def all_img_paths(path):
    all_img_paths = glob.glob(os.path.join(path, "*", "*"))
    all_img_paths = [p for p in all_img_paths if os.path.isfile(p)]
    y = [os.path.basename(os.path.dirname(p)) for p in all_img_paths]
    return all_img_paths,y

def get_img(p):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Image not found or unable to read: {p}")
        try:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype('float64')
        except cv2.error:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float64')
        img = img / 255.
        img = cv2.resize(img, (256, 256))
        return img

def pos_list(n=8,l=52):
    '''
    Usage:
        Create the desired position of CGM 
    n--> Number of Images required
    l ---> Number of Currently available images
    skip ---> interval between two consecutive picks

    formula = summation ceil(ceil(skip*i)*(l/100))
    '''
    skip=(100/(n-1))
    print(skip)
    percentages=[math.ceil(skip*i) for i in range(n)]

    one_percent =l/100
    print(one_percent)
    return [math.ceil(i*one_percent) for i in percentages]