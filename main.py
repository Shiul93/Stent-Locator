from colors import bcolors
print bcolors.HEADER+ 'Loading imutils'
import imutils as imutils

print 'Loading opencv'
import cv2
print 'Loading numpy'

import numpy as np
from functions import *


import argparse

print 'Loading snakes'+bcolors.ENDC
import morphsnakes

parser = argparse.ArgumentParser(description='Strut location')
parser.add_argument('input', metavar='F', type=str,
                    help='Input file route')

# Parse arguments
args = parser.parse_args()
input_path = args.input

# Load image
im = cv2.imread(input_path)

print 'Loaded image '+input_path

cv2.namedWindow("window")

# Clean image
cropped = removeThings(im)

# Convert image to gray scale
imbw = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

# Clean the center of the image
imbw = removeCenter((imbw))

output = imbw * 1


centerx = 351
centery = 263
radius = 50





# Find the contour and the surface of the artery using active contours

img = imbw/255.0
gI = morphsnakes.gborders(img, alpha=3000, sigma=5.48)
mgac = morphsnakes.MorphGAC(gI, smoothing=1, threshold=0.31, balloon=1)
mgac.levelset = circle_levelset(img.shape, (centery,centerx), radius)

mask, edges = morphsnakes.evolve(mgac, num_iters=170, animate=True, background=imbw)


area(mask)



pixelarea, totalarea = area(mask)

print bcolors.OKBLUE+'Total area occuppied by aorta section: '+str(totalarea)+bcolors.ENDC
print bcolors.OKBLUE+'Number of pixels of the aorta section: '+str(pixelarea)+bcolors.ENDC





ret,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
im2,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
M = cv2.moments(cnt)

cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

image = imbw*1
cv2.drawContours(image, [cnt], -1, 255, 1)
cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
cv2.putText(image, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.imshow("window", image)
cv2.waitKey(500)

image = imbw*1


(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(image,center,radius,255,2)
cv2.imshow('window',image)
cv2.waitKey(500)

image = imbw*0
cv2.circle(image,center,radius,255,cv2.FILLED)

circularity = (len(np.nonzero(mask)[0])/float(len(np.nonzero(image)[0])))

print bcolors.OKBLUE+'Aorta circularity: '+str(circularity)+bcolors.ENDC
print bcolors.OKBLUE+'Aorta radius: '+str(radius)+bcolors.ENDC


x,y,w,h = cv2.boundingRect(cnt)
print bcolors.OKBLUE+'Aorta aspect ratio: '+str(w)+':'+str(h)+bcolors.ENDC


cv2.waitKey(500)


polar = cv2.linearPolar(imbw,center,526/2,cv2.INTER_NEAREST)
polarmask = cv2.linearPolar(mask,center,526/2,cv2.INTER_NEAREST)
#polar = img2polar(imbw,center,526/2,526/2)

print 'Polar: '+ str(polar.shape)
print 'Original: '+str(imbw.shape)
polar = removeNoise(polar)
cv2.imshow('window',polar)
cv2.waitKey(500)
cv2.imshow('window',polarmask)
cv2.waitKey(500)


smask = stentMask(polar)
cv2.imshow('window',smask)
cv2.waitKey(500)

ret,thresh = cv2.threshold(polar,25 ,255,cv2.THRESH_BINARY)
cv2.imshow('window', thresh)
cv2.waitKey(500)


stents = (thresh/255 * smask/255)*255
cv2.imshow('window', stents)
cv2.waitKey(500)



stents = (stents/255*(shiftImage(polarmask,15)/255))*255
cv2.imshow('window', stents)
cv2.waitKey(500)

cv2.imshow("window", cv2.addWeighted(polar,0.5,stents,0.5,0.0))
cv2.waitKey(500)

new = cv2.linearPolar(stents,center,526/2,flags = (cv2.WARP_INVERSE_MAP))

cv2.imshow("window", cv2.addWeighted(imbw,0.5,new,0.5,0.0))
cv2.waitKey(0)
