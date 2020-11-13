import numpy as np
import cv2
import glob

path = 'files/test_calib_pics_l/*.bmp'
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(path)
teller = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (6,4),None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (6,4), corners,ret)

        #cv2.imshow('img',img)
        #cv2.waitKey(1000)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(mtx)

with open('camera_l_intr_original.txt', 'wb') as f:
    np.savetxt(f, mtx)

with open('camera_l_dist_original.txt','wb') as f:
    np.savetxt(f,dist)


path2 = 'files/test_calib_pics_l/1603384357.301529.bmp'

img_org = cv2.imread(path2)
h,  w = img_org.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

with open('camera_l_intr_optimized.txt', 'wb') as f:
    np.savetxt(f, newcameramtx)


print(roi)
print(newcameramtx)
cv2.destroyAllWindows()