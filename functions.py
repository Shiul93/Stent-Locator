from random import randrange

import numpy as np
import cv2

"""
Removes estructures from input image
"""


# noinspection PyTypeChecker
def removeThings(input_img):
    print 'Clean Image'
    img = input_img.copy()
    roi = [1, 526, 2, 703]
    cropped = img[1:526, 2:703]
    cropped[478:518, 658:696] = 0
    c1 = [255, 255, 255]
    c2 = [0, 255, 255]
    matches_not = np.logical_not(np.logical_or(np.all(cropped == c1, -1), np.all(cropped == c2, -1)))
    matches = (np.logical_or(np.all(cropped == c1, -1), np.all(cropped == c2, -1)))


    cropped[:, :, 0] = cropped[:, :, 0] * matches_not.astype(int)
    cropped[:, :, 1] = cropped[:, :, 1] * matches_not.astype(int)
    cropped[:, :, 2] = cropped[:, :, 2] * matches_not.astype(int)

    kernel = np.ones((9, 9), np.float32) / 81

    # filtered = cv2.filter2D(cropped, -1, kernel)
    filtered = cv2.medianBlur(cropped, 9)
    filtered[:, :, 0] = filtered[:, :, 0] * matches
    filtered[:, :, 1] = filtered[:, :, 1] * matches
    filtered[:, :, 2] = filtered[:, :, 2] * matches

    out = cropped + filtered
    print 'Clean done'
    return out

'''
Removes the center estructure from a bw image
'''
def removeCenter(bwimage):
    output = bwimage*1
    mask = bwimage*0
    cv2.circle(output, (351, 263), 50, 0, -1)
    cv2.circle(mask, (351, 263), 50, 1, -1)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y]==1:
                mask[x,y]=randrange(0,12)

    output = output+mask
    return output



def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u


def getEdge(img):
    kernel = np.ones((3, 3))
    erosion = cv2.erode(img, kernel)

    edges = (img - erosion).astype(np.uint8) * 255

    return edges


def area(img):
    nonzero = len((np.nonzero(img))[0])
    totalarea = img.shape[0]*img.shape[1]

    return nonzero, (nonzero/float(totalarea))

def removeNoise(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y]<14:
                img[x,y]=0
    return img


def stentMask(img):
    mean = np.mean(img)
    output = img*0
    for x in range(img.shape[0]):
        if np.mean(img[x,:])< (mean*0.5):
            output[x,:]=255

    return output

def shiftImage(img, pixels):
    output = img*1
    output[:,pixels:]= img[:,:-pixels]
    return output



