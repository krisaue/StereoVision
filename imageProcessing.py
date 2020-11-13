"""
Author: Kristian Auestad
Date:   24.10.2020
Name: ImageProcessing.py
Description: Collection of general functions used to preprocess images.
"""

import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
#local
import feature_extractor
from functions import *

### LOAD PARAMETERS ###
c_l_intr_org = np.loadtxt('files/cam_params/camera_l_intr_original.txt')
c_l_intr_opt = np.loadtxt('files/cam_params/camera_l_intr_optimized.txt')
c_l_dist = np.loadtxt('files/cam_params/camera_l_dist_original.txt')

c_r_intr_org = np.loadtxt('files/cam_params/camera_r_intr_original.txt')
c_r_intr_opt = np.loadtxt('files/cam_params/camera_r_intr_optimized.txt')
c_r_dist = np.loadtxt('files/cam_params/camera_r_dist_original.txt')

def getOptIntrMatrix(intr_mtx, dist_coeff, img_path):
    img = cv.imread(img_path)
    h,w = img.shape[:2]
    newIntrMatrix, roi = cv.getOptimalNewCameraMatrix(intr_mtx, dist_coeff,(w,h),1,(w,h))
    return newIntrMatrix,roi

def loadImg(path):
    img = cv.imread(path)
    return img

def undistort(img_path,intr_matrix,dist_coeffs):
    newIntrMatrix,roi = getOptIntrMatrix(intr_matrix,dist_coeffs,img_path)
    img = cv.imread(img_path)
    dst = cv.undistort(img,intr_matrix,dist_coeffs,None,newIntrMatrix)
    x, y, w, h = roi
    h = 889
    w = 1095
    dst = dst[y:y+h, x:x+w]
    return dst

def detectFeatures(img):
    #img = cv.imread(img_path)
    
    extr = feature_extractor.Exctractor(2)
    
    kp,des,feature = extr.extract_feature(img)
    return kp, des,feature
    
BF = 0
FLANN = 1

def matchFeatures(masterDescriptors,slaveDescriptors,matcherType=FLANN,matchPercent=0.15):
    if matcherType == BF:
        match_method = cv.DescriptorMatcher_create(
            cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    elif matcherType == FLANN:
        match_method = cv.FlannBasedMatcher_create()

    matches = match_method.match(masterDescriptors, slaveDescriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    numGoodMatches = int(len(matches) * matchPercent)
    matches = matches[:numGoodMatches]
    return matches

def triangulatePoints(match):
    pts1 = np.zeros((len(match['indexes']), 2), dtype=np.float32)
    pts2 = np.zeros((len(match['indexes']), 2), dtype=np.float32)
    des1 = np.zeros((len(match['indexes']), 128), dtype=np.float32)
    des2 = np.zeros((len(match['indexes']), 128), dtype=np.float32)
    for i, indexes in enumerate(match['indexes']):
        pts1[i, :] = q_kp[indexes[0]].pt
        pts2[i, :] = t_kp[indexes[1]].pt
        des1[i, :] = q_des[indexes[0]]
        des2[i, :] = t_des[indexes[1]]

    point_matches = np.column_stack((pts1, pts2))
    return point_matches


##CONFIG##
loadImgs = True
extractFeatures = False
matchFeat = False
rectImg = True
SGM123 = False
testing = False


if __name__ == "__main__":
    if loadImgs:
        img_l_path = 'files/test_pics/test2_l.bmp'
        img_r_path = 'files/test_pics/test2_r.bmp'
        
        img_l = loadImg(img_l_path)
        img_r = loadImg(img_r_path)

        img_l_undist = undistort(img_l_path,c_l_intr_org,c_l_dist)
        img_r_undist = undistort(img_r_path,c_r_intr_org,c_r_dist)
    if extractFeatures:
        kp_l,des_l,feature_l = detectFeatures(img_l_undist)
        kp_r,des_r,feature_r = detectFeatures(img_r_undist)
    
    if matchFeat:
        matches = matchFeatures(des_l,des_r,matcherType=BF)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv.drawMatches(img_l_undist,kp_l,img_r_undist,kp_r,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()
    if rectImg:
        img_size = [889,1095]
        R1,R2,P1,P2,roi_l,roi_r = rectifyImage(c_l_intr_org,c_r_intr_org,c_l_dist,c_r_dist,img_size)
        
        map1_l,map2_l = cv.initUndistortRectifyMap(c_l_intr_org,c_l_dist,R1,P1,(img_size[0],img_size[1]),cv.CV_32FC2)

        print("map",map2_l)
        l_dst = cv.remap(img_l,map1_l,map2_l,cv.INTER_LINEAR)
        plt.imshow(img_l)
        plt.imshow(l_dst)


    if SGM123:
        SGM(img_l_undist,img_r_undist)
    if testing:
        m_idx = []
        s_idx = []
        for k,points in enumerate(matches):
            m_idx.append(matches[k].queryIdx)
            s_idx.append(matches[k].trainIdx)

        m_match_points = np.zeros((len(m_idx),2))
        s_match_points = np.zeros((len(m_idx),2))

        for i in range(len(m_idx)):
            m_match_points[i,0] = (kp_l[m_idx[i]].pt[0])
            m_match_points[i,1] = (kp_l[m_idx[i]].pt[1])

            s_match_points[i,0] = (kp_r[s_idx[i]].pt[0])
            s_match_points[i,1] = (kp_r[s_idx[i]].pt[1])

        essMat = cv.findEssentialMat(m_match_points,s_match_points,c_l_intr_opt,method=8,prob=0.999,threshold=1)
        Rts = motion_from_essential(essMat[0])
        #print(Rts)
        Rts_best = choose_solution(m_match_points,s_match_points,c_l_intr_opt,c_r_intr_opt,Rts)
        print(Rts_best)