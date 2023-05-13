import pandas as pd
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

def gnoise_lincontrast(im_tr, pt_tr):
  seq = iaa.Sequential([iaa.LinearContrast((0.6, 1.5)), 
                        iaa.Sometimes(
        0.80, iaa.GaussianBlur(sigma=(0., 2.0)))])
  aug_ims = []
  aug_pts = []
  for im, pt in zip(im_tr, pt_tr):
    #f_im, f_pts = flip_im_points1(im, pt)
    f_im = seq(image=im)
    aug_ims.append(im)
    aug_ims.append(f_im)
    aug_pts.append(pt)
    aug_pts.append(pt)
  return np.asarray(aug_ims), np.asarray(aug_pts)

def flip_im_points(img, points): # flip keypoints when they are not standardized 
  flip_im = np.fliplr(img)
  xcoords = points[0::2]
  ycoords = points[1::2]
  new_points = []
  for i in range(len(xcoords)):
    xp = xcoords[i]
    yp = ycoords[i]
    new_points.append(96-xp)
    new_points.append(yp)
  return flip_im, np.asarray(new_points) 

def aug_flip(im_tr, pt_tr):
  aug_ims = []
  aug_pts = []
  for im, pt in zip(im_tr, pt_tr):
    f_im, f_pts = flip_im_points(im, pt)
    aug_ims.append(im)
    aug_ims.append(f_im)
    aug_pts.append(pt)
    aug_pts.append(f_pts)
  return np.asarray(aug_ims), np.asarray(aug_pts)

def rotate_scale_aug(im_tr, pt_tr):
  seq = iaa.Sequential([iaa.Affine(rotate=15, scale=(0.8, 1.2))])
  #image_aug, kps_aug = seq(image=image, keypoints=kps)
  aug_ims = []
  aug_pts = []
  coordlist = []
  for im, pt in zip(im_tr, pt_tr):
    #f_im, f_pts = flip_im_points1(im, pt)
    xcoord = pt[0::2]
    ycoord = pt[1::2]
    for i in range(len(xcoord)): 
      coordlist.append(Keypoint(xcoord[i], ycoord[i]))
    kps = KeypointsOnImage(coordlist, shape=im.shape)  
    f_im, f_kp = seq(image=im, keypoints=kps)
    #new_xcoords = []
    #new_ycoords = []
    all_coords = []
    for k in range(len(kps.keypoints)):
      before = kps.keypoints[k]
      after = f_kp.keypoints[k]
      # print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
      #     i, before.x, before.y, after.x, after.y)
      # )
      all_coords.append(after.x)
      all_coords.append(after.y)
      all_coords_arr = np.asarray(all_coords)
    aug_ims.append(im)
    aug_ims.append(f_im)
    aug_pts.append(pt)
    aug_pts.append(all_coords)
    coordlist.clear()
  return np.asarray(aug_ims), np.asarray(aug_pts)