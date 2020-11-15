import numpy as np
import cv2 as cv

def linear_triangulation(uv1, uv2, P1, P2):
    """
    Compute the 3D position of a single point from 2D correspondences.
    Args:
        uv1:    2D projection of point in image 1.
        uv2:    2D projection of point in image 2.
        P1:     Projection matrix with shape 3 x 4 for image 1.
        P2:     Projection matrix with shape 3 x 4 for image 2.
    Returns:
        X:      3D coordinates of point in the camera frame of image 1.
                (not homogeneous!)
    See HZ Ch. 12.2: Linear triangulation methods (p312)
    """

    A = np.row_stack((
        uv1[0]*P1.T[:, 2] - P1.T[:, 0],
        uv1[1]*P1.T[:, 2] - P1.T[:, 1],
        uv2[0]*P2.T[:, 2] - P2.T[:, 0],
        uv2[1]*P2.T[:, 2] - P2.T[:, 1],
    ))

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    X = np.resize(vh[-1, :], (4, 1))
    X = X/X[len(X)-1]
    X = np.resize(X, (3, 1))
    X = np.ndarray.flatten(X)
    return X

def motion_from_essential(E):
    """ Computes the four possible decompositions of E into
    a relative rotation and translation.
    See HZ Ch. 9.7 (p259): Result 9.19
    """
    U, s, VT = np.linalg.svd(E)

    # Make sure we return rotation matrices with det(R) == 1
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(VT) < 0:
        VT = -VT

    W = np.array([[0, -1, 0], [+1, 0, 0], [0, 0, 1]])
    R1 = U@W@VT
    R2 = U@W.T@VT
    t1 = U[:, 2]
    t2 = -U[:, 2]
    return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]


def choose_solution(uv1, uv2, K1, K2, Rts):
    """
    Chooses among the rotation and translation solutions Rts
    the one which gives the most points in front of both cameras.
    """
    n = len(uv1)
    best = (0, 0)
    for i, (R, t) in enumerate(Rts):
        P1, P2 = camera_matrices(K1, K2, R, t)
        X1 = np.array([linear_triangulation(uv1[j], uv2[j], P1, P2)
                       for j in range(n)])
        X2 = X1 @ R.T + t
        visible = np.logical_and(X1[:, 2] > 0, X2[:, 2] > 0)
        num_visible = np.sum(visible)
        if num_visible > best[1]:
            best = (i, num_visible)
    #print('Choosing solution %d (%d points visible)' % (best[0], best[1]))
    return Rts[best[0]]


def camera_matrices(K1, K2, R, t):
    """ Computes the projection matrix for camera 1 and camera 2.
    Args:
        K1,K2: Intrinsic matrix for camera 1 and camera 2.
        R,t: The rotation and translation mapping points in camera 1 to points in camera 2.
    Returns:
        P1,P2: The projection matrices with shape 3x4.
    """
    P1 = K1@np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K2@np.column_stack((R, t))
    return P1, P2


def linear_triangulation(uv1, uv2, P1, P2):
    """
    Compute the 3D position of a single point from 2D correspondences.
    Args:
        uv1:    2D projection of point in image 1.
        uv2:    2D projection of point in image 2.
        P1:     Projection matrix with shape 3 x 4 for image 1.
        P2:     Projection matrix with shape 3 x 4 for image 2.
    Returns:
        X:      3D coordinates of point in the camera frame of image 1.
                (not homogeneous!)
    See HZ Ch. 12.2: Linear triangulation methods (p312)
    """
    A = np.empty((4, 4))
    A[0, :] = uv1[0]*P1[2, :] - P1[0, :]
    A[1, :] = uv1[1]*P1[2, :] - P1[1, :]
    A[2, :] = uv2[0]*P2[2, :] - P2[0, :]
    A[3, :] = uv2[1]*P2[2, :] - P2[1, :]
    U, s, VT = np.linalg.svd(A)
    X = VT[3, :]
    return X[:3]/X[3]


def epipolar_match(I1, I2, F, uv1):
    """
    For each point in uv1, finds the matching point in image 2 by
    an epipolar line search.
    Args:
        I1:  (H x W matrix) Grayscale image 1
        I2:  (H x W matrix) Grayscale image 2
        F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
        uv1: (n x 2 array) Points in image 1
    Returns:
        uv2: (n x 2 array) Best matching points in image 2.
    """

    # Tips:
    # - Image indices must always be integer.
    # - Use int(x) to convert x to an integer.
    # - Use rgb2gray to convert images to grayscale.
    # - Skip points that would result in an invalid access.
    # - Use I[v-w : v+w+1, u-w : u+w+1] to extract a window of half-width w around (v,u).
    # - Use the np.sum function.

    w = 10
    uv2 = np.zeros(uv1.shape)
    for i, (u1, v1) in enumerate(uv1):
        if u1 < w or v1 < w or u1 > I1.shape[1]-w or v1 > I1.shape[0]-w:
            continue
        l = F@np.array((u1, v1, 1))
        W1 = I1[int(v1)-w:int(v1)+w+1, int(u1)-w:int(u1)+w+1]

        best_err = np.inf
        best_u2 = w
        for u2 in range(w, I2.shape[1]-w):
            v2 = int(round(-(l[2] + u2*l[0])/l[1]))
            if v2 < w or v2 > I2.shape[0]-w:
                continue
            W2 = I2[v2-w:v2+w+1, u2-w:u2+w+1]
            err = np.sum(np.absolute(W1 - W2))
            if err < best_err:
                best_err = err
                best_u2 = u2

        uv2[i, 0] = best_u2
        uv2[i, 1] = -(l[2] + best_u2*l[0])/l[1]
    return uv2


def rgb2gray(I):
    """
    Converts a red-green-blue (RGB) image to grayscale brightness.
    """
    return 0.2989*I[:, :, 0] + 0.5870*I[:, :, 1] + 0.1140*I[:, :, 2]


def get_intrinsics(imName):

    if len(imName) > 15:
        K = np.loadtxt('calibration/intr_nok7.txt')
    else:
        K = np.loadtxt('calibration/intr_iph7.txt')

    return K



def SGM(imgL,imgR):
    # disparity range is tuned for 'aloe' image pair
    imgL = cv.pyrDown(imgL)  # downscale images for faster processing
    imgR = cv.pyrDown(imgR)
   
   
    window_size = 8
    min_disp = -10
    num_disp = 32
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 6,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 200,
        speckleRange = 1
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 1000                         # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)

    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    #write_ply(out_fn, out_points, out_colors)
    #print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')

def rectifyImage(intr_mat_l,intr_mat_r,dist_coeffs_l,dist_coeffs_r,img_size):
    beta = np.deg2rad(2)
    R1 = np.array([[np.cos(beta),0,-np.sin(beta)],[0,1,0],[np.sin(beta),0,np.cos(beta)]])
    t = np.array([-1800.0,0.0,0.0]).T
    print(t.shape)
    
    R1, R2, P1, P2, Q, roi_left, roi_right = cv.stereoRectify(intr_mat_l, dist_coeffs_l,intr_mat_r,dist_coeffs_r,(img_size[0],img_size[1]),R1,t)
    #ans = cv.stereoRectify(intr_mat_l,dist_coeffs_l,intr_mat_r,dist_coeffs_r,(img_size[0],img_size[1]),R1,t)

    #for k,elem in enumerate(ans):
    #    print("ANS[] : {0}".format(ans[k]))



    return R1,R2,P1,P2, roi_left,roi_right


