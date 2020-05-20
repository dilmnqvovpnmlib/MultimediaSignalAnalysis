import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import display, Image

def get_image_data(im):
    rgb = np.array(im)
    return im, rgb

def add_edge(img):
    H, W = img.shape
    WID = 1000# int(np.max(img.shape) * 2 ** 0.5)
    print(WID)
    e_img = np.zeros((WID, WID))
    e_img[
        int((WID-H)/2):int((WID+H)/2),
        int((WID-W)/2):int((WID+W)/2)
    ] = img
    return e_img

if __name__ == '__main__':
    path_src = './input/1-004-1.jpg'
    inputs_src = cv2.imread(path_src)
    edges_src = cv2.cvtColor(inputs_src, cv2.COLOR_RGB2GRAY)
    output = add_edge(edges_src)
    cv2.imwrite("test1.png", output)