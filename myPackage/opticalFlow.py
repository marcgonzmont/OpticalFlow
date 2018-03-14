import numpy as np
import numpy.linalg as nlin
import cv2
from numba import jit


@jit
def getDerivative(img1, img2, k_gauss):
    '''

    :param img1:
    :param img2:
    :param k_gauss:
    :return:
    '''

    # Gaussian filter to blur the images and improve temporal derivative
    img1_gauss = cv2.GaussianBlur(img1, k_gauss, 0)
    img2_gauss = cv2.GaussianBlur(img2, k_gauss, 0)

    # IMG1
    dx_1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0)
    dy_1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1)

    # IMG2
    dx_2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0)
    dy_2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1)

    # Get the mean of the derives and the temporal derivative
    dx = (dx_1*0.5 + dx_2*0.5)
    dy = (dy_1*0.5 + dy_2*0.5)
    dt = img2_gauss - img1_gauss

    return dx, dy, dt


@jit
def multiplyMatrix(dx, dy, dt):
    '''

    :param dx:
    :param dy:
    :param dt:
    :return:
    '''

    Ix_2 = np.multiply(dx, dx)
    Iy_2 = np.multiply(dy, dy)
    Ixy_2 = np.multiply(dx, dy)
    Ixt_2 = np.multiply(dx, dt)
    Iyt_2 = np.multiply(dy, dt)

    return Ix_2, Iy_2, Ixy_2, Ixt_2, Iyt_2


@jit
def multiplyMatrix2(Ix2, Iy2, Ixy, Ixt, Iyt):
    M0 = (-1) * Ixt * Iy2
    M1 = Ixy * Iyt
    M2 = Ix2 * Iy2
    M3 = Ixy * Ixy
    M4 = Ixy * Ixt
    M5 = Ix2 * Iyt

    return M0, M1, M2, M3, M4, M5


# @jit
def initVariables(im_shape, k_gauss, n_iter, lam_pond, dx, dy, dt):
    u_m = np.zeros(im_shape)
    v_m = np.zeros(im_shape)

    for it in range(n_iter):
        u_m = cv2.GaussianBlur(u_m, k_gauss, 0)
        v_m = cv2.GaussianBlur(v_m, k_gauss, 0)
        ratio = ((np.multiply(dx, u_m) + np.multiply(dy, v_m) + dt) / (lam_pond**2 + dx**2 + dy**2))
        u_m = u_m - (dx * ratio)
        v_m = v_m - (dy * ratio)

    return u_m, v_m


def draw_hsv(fx, fy):
    h, w = fx.shape
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


# @jit
def computeOF_LK_pinv(fr1, fr2, window, k_gauss):
    '''

    :param fr1:
    :param fr2:
    :param window:
    :param A:
    :param B:
    :param k_gauss:
    :return:
    '''
    img1 = fr1 * 1 / 255
    img2 = fr2 * 1 / 255
    # u = np.zeros_like(img1)
    # v = np.zeros_like(img1)
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]
    step = int(np.floor(window / 2))

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivative(img1, img2, k_gauss)
    Ix_2, Iy_2, Ixy_2, Ixt_2, Iyt_2 = multiplyMatrix(dx, dy, dt)

    # h_lim = int(np.ceil(h / window))
    # w_lim = int(np.ceil(w / window))

    for i in range(step, h - step, 1):
        for j in range(step, w - step, 1):
            # Summation
            Ix2 = np.sum(Ix_2[i - step : i + step, j - step : j + step])
            Iy2 = np.sum(Iy_2[i - step : i + step, j - step : j + step])
            Ixy = np.sum(Ixy_2[i - step : i + step, j - step : j + step])
            Ixt = -np.sum(Ixt_2[i - step : i + step, j - step : j + step])
            Iyt = -np.sum(Iyt_2[i - step : i + step, j - step : j + step])

            # Initialization of matrix A
            A[0, 0] = Ix2
            A[1, 0] = A[0, 1] = Ixy
            A[1, 1] = Iy2

            pinv_A = np.matrix.round(nlin.pinv(A))

            # Initialization of matrix B
            b[0, 0] = Ixt
            b[1, 0] = Iyt

            # Set u, v vector
            u, v = np.matrix.round(np.dot(pinv_A, b))

            if i % step == 0 and j % step == 0:
                cv2.arrowedLine(img2, (j, i), (j + u, i + v), (255, 255, 0))
                cv2.arrowedLine(optical_flow, (j, i), (j + u, i + v), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow("Optical flow", result)
    cv2.waitKey(50)
    # cv2.destroyAllWindows()

    return result


'''
@jit
def computeOF_OCV(fr1, fr2, window):
    optical_flow = np.zeros_like(fr1)
    optical_flow[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(fr1, fr2, None, 0, 1, window, 1, 1, 1, 1)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    optical_flow[...,0] = ang*180/np.pi/2
    optical_flow[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    optical_flow = cv2.cvtColor(optical_flow,cv2.COLOR_HSV2BGR)

    result = np.concatenate((fr2, optical_flow), axis= 1)

    cv2.imshow("Optical flow", result)
    cv2.waitKey()

    return result
'''


@jit
def computeOF_LK_unrolled(fr1, fr2, window, step, k_gauss):
    img1 = fr1 * 1/255
    img2 = fr2 * 1 / 255
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivative(img1, img2, k_gauss)
    Ix_2, Iy_2, Ixy_2, Ixt_2, Iyt_2 = multiplyMatrix(dx, dy, dt)

    h_lim = int(np.ceil(h / window - 1))
    w_lim = int(np.ceil(w / window - 1))

    for i in range(h_lim):
        for j in range(w_lim):
            ii = i * window + step
            jj = j * window + step

            # Summation
            Ix2 = np.sum(Ix_2[ii - step : ii + step, jj - step : jj + step])
            Iy2 = np.sum(Iy_2[ii - step : ii + step, jj - step : jj + step])
            Ixy = np.sum(Ixy_2[ii - step : ii + step, jj - step : jj + step])
            Ixt = -np.sum(Ixt_2[ii - step : ii + step, jj - step : jj + step])
            Iyt = -np.sum(Iyt_2[ii - step : ii + step, jj - step : jj + step])

            M0, M1, M2, M3, M4, M5 = multiplyMatrix2(Ix2, Iy2, Ixy, Ixt, Iyt)

            u = (M0 + M1) / (M2 - M3)
            v = (M4 + M5) / (M2 - M3)

            cv2.arrowedLine(img2, (jj, ii), (int(jj + u), int(ii + v)), (255, 255, 0))
            cv2.arrowedLine(optical_flow, (jj, ii), (int(jj + u), int(ii + v)), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow("Optical flow", result)
    cv2.waitKey(50)

    return result


@jit
def computeOF_HS(fr1, fr2, window, step, k_gauss, n_iter, lam_pond):
    img1 = fr1 * 1 / 255
    img2 = fr2 * 1 / 255
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivative(img1, img2, k_gauss)

    h_lim = int(np.ceil(h / window - 1))
    w_lim = int(np.ceil(w / window - 1))
    im_shape = img1.shape

    u_m, v_m = initVariables(im_shape, k_gauss, n_iter, lam_pond, dx, dy, dt)

    for i in range(h_lim):
        for j in range(w_lim):
            ii = i * window + step
            jj = j * window + step

            u = np.sum(np.sum(u_m[ii - step : ii + step, jj - step : jj + step]))
            v = np.sum(np.sum(v_m[ii - step: ii + step, jj - step: jj + step]))

            u_i = int(u)
            v_i = int(v)

            cv2.arrowedLine(img2, (jj, ii), (int(jj + u_i), int(ii + v_i)), (255, 255, 0))
            cv2.arrowedLine(optical_flow, (jj, ii), (int(jj + u_i), int(ii + v_i)), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow('Optical flow', result)
    cv2.waitKey(50)

    return result