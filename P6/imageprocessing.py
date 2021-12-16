"""Image processing"""
import pandas as pd
import numpy as np

import PIL.ImageOps as imageops
from PIL import Image

import skimage
from skimage import io
from skimage import restoration

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ast import literal_eval

import cv2

import missingno as msno
from tqdm import tqdm

# visualisation des categories dans un dataframe
def category_trees(data):
    
    prod_cat_lists = []
    product_trees = (data.product_category_tree.apply(lambda x: x.replace(' >> ', '","'))
                     .apply(literal_eval)
                    )
    
    for tree in product_trees:
        prod_cat_lists.append(tree)
    
    df = pd.DataFrame(prod_cat_lists)
    
    
    return df


# création de la variable des catégories de produits
def extract_categories_from_tree(data, level=0):
    
    df = category_trees(data)

    return df[level]



def load_images(filelist, path='./data/Flipkart/Images/'):

    images = []

    for i, l in enumerate(tqdm(filelist)):
        # chemin vers l'image
        filename = path + l
        images.append(mpimg.imread(filename))

    
    return np.asarray(images, dtype='object')


def reduce_image_size(image, image_ind, data, write=False, **kwargs):
    
    # calculate new dimensions
    orig_width = image.shape[1]
    orig_height = image.shape[0]
    
    
    max_width = kwargs.pop("max_width", None)
    max_height = kwargs.pop("max_height", None)
    scale_factor = kwargs.pop("scale_factor", None)
     
    if max_width and max_height:
        dim_array = np.array(
            [(orig_width, max_width),
            (orig_height, max_height)]
        )
    
        if dim_array[0, 0] == dim_array[1, 0]:
            largest_dim = np.argmax(dim_array, axis=0)[1]
        else:
            largest_dim = np.argmax(dim_array, axis=0)[0]
    
        scale_dim = dim_array[largest_dim, 1]
        scale_factor = dim_array[largest_dim, 0]/scale_dim
    else:
        if not scale_factor:
            scale_factor = 1
        
    width = int(image.shape[1] / scale_factor)
    height = int(image.shape[0] / scale_factor)

    # dsize
    dsize = (width, height)
    
    # resize
    img = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    
    if write:
        # save file to directory
        filename = data.image.iloc[image_ind]
        path_to_write = kwargs.pop("path") + filename
        cv2.imwrite(path_to_write, img)
    
    return img
    
    


def custom_autocontrast(image, cutoff=0, ignore=None, mask=None, preserve_tone=False):
    """Code adapted from PIL.ImageOps.autocontrast to work with non PIL.Image objects"""
    """
    Maximize (normalize) image contrast. This function calculates a
    histogram of the input image (or mask region), removes ``cutoff`` percent of the
    lightest and darkest pixels from the histogram, and remaps the image
    so that the darkest pixel becomes black (0), and the lightest
    becomes white (255).

    :param image: The image to process.
    :param cutoff: The percent to cut off from the histogram on the low and
                   high ends. Either a tuple of (low, high), or a single
                   number for both.
    :param ignore: The background pixel value (use None for no background).
    :param mask: Histogram used in contrast operation is computed using pixels
                 within the mask. If no mask is given the entire image is used
                 for histogram computation.
    :param preserve_tone: Preserve image tone in Photoshop-like style autocontrast.

                          .. versionadded:: 8.2.0

    :return: An image.
    """
    #if preserve_tone:
        
    # histogram = image.convert("L").histogram(mask)
    img = Image.fromarray(image)
    histogram = img.histogram(mask)

    lut = []
    for layer in range(0, len(histogram), 256):
        h = histogram[layer : layer + 256]
        if ignore is not None:
            # get rid of outliers
            try:
                h[ignore] = 0
            except TypeError:
                # assume sequence
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = n * cutoff[0] // 100
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the high end
            cut = n * cutoff[1] // 100
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut.extend(list(range(256)))
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                ix = int(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut.append(ix)
    
    return np.asarray(img.point(lut))

def image_preprocessing(im):
    """Implements the following preprocessing steps:
    - change color space from RGB to YCrCb
    - stretches (normalizes) histogram on the Y channel
    - equalize histogram on the Y channel
    - change color space back to RGB
    
    After completion of the preprocessing steps, returns the image"""
    
    # convert image from RGB to YCrCb
    im_ycc = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)

    # étirement de l'histogramme sur le canal Y
    im_ycc[:, :, 0] = custom_autocontrast(im_ycc[:, :, 0], cutoff=(0.025, 0))
    
    # Histogram equalisation on the Y-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_ycc[:, :, 0] = clahe.apply(im_ycc[:, :, 0])

    # convert image back from YCrCb to BGR
    img = cv2.cvtColor(im_ycc, cv2.COLOR_YCrCb2RGB)
    
    return  img