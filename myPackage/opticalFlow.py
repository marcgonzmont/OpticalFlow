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


@jit
def computeOF_LK_pinv(fr1, fr2, window, k_gauss, plt_step):
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
    A = np.zeros((2,2))
    b = np.zeros((2,1))
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]
    step = int(np.floor(window / 2))

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivative(img1, img2, k_gauss)
    Ix_2, Iy_2, Ixy_2, Ixt_2, Iyt_2 = multiplyMatrix(dx, dy, dt)

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

            if i % plt_step == 0 and j % plt_step == 0:
                cv2.arrowedLine(img2, (j, i), (j + u, i + v), (255, 255, 0))
                cv2.arrowedLine(optical_flow, (j, i), (j + u, i + v), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow("Optical flow Lucas-Kanade (pinv)", result)
    cv2.waitKey(50)
    # cv2.destroyAllWindows()

    return result

@jit
def computeOF_LK_unrolled(fr1, fr2, window, k_gauss, plt_step):
    img1 = fr1 * 1 / 255
    img2 = fr2 * 1 / 255
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]
    step = int(np.floor(window / 2))

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivative(img1, img2, k_gauss)
    Ix_2, Iy_2, Ixy_2, Ixt_2, Iyt_2 = multiplyMatrix(dx, dy, dt)

    for i in range(step, h - step, 1):
        for j in range(step, w - step, 1):
            # Summation
            Ix2 = np.sum(Ix_2[i - step: i + step, j - step: j + step])
            Iy2 = np.sum(Iy_2[i - step: i + step, j - step: j + step])
            Ixy = np.sum(Ixy_2[i - step: i + step, j - step: j + step])
            Ixt = -np.sum(Ixt_2[i - step: i + step, j - step: j + step])
            Iyt = -np.sum(Iyt_2[i - step: i + step, j - step: j + step])

            M0, M1, M2, M3, M4, M5 = multiplyMatrix2(Ix2, Iy2, Ixy, Ixt, Iyt)
            div = (M2 - M3)

            if div != 0:
                u = int(np.floor((M0 + M1) / (M2 - M3)))
                v = int(np.floor((M4 + M5) / (M2 - M3)))
            else:
                continue

            if i % plt_step == 0 and j % plt_step == 0:
                cv2.arrowedLine(img2, (j, i), (j + u, i + v), (255, 255, 0))
                cv2.arrowedLine(optical_flow, (j, i), (j + u, i + v), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow("Optical flow Lucas-Kanade (unrolled)", result)
    cv2.waitKey(50)

    return result


@jit
def computeOF_HS(fr1, fr2, window, k_gauss, n_iter, lam_pond, plt_step):
    img1 = fr1 * 1 / 255
    img2 = fr2 * 1 / 255
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]
    step = int(np.floor(window / 2))

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivative(img1, img2, k_gauss)

    im_shape = img1.shape

    u_m, v_m = initVariables(im_shape, k_gauss, n_iter, lam_pond, dx, dy, dt)

    for i in range(step, h - step, 1):
        for j in range(step, w - step, 1):

            u = np.sum(np.sum(u_m[i - step : i + step, j - step : j + step]))
            v = np.sum(np.sum(v_m[i - step: i + step, j - step: j + step]))

            u_i = int(u)
            v_i = int(v)

            if i % plt_step == 0 and j % plt_step == 0:
                cv2.arrowedLine(img2, (j, i), (j + u_i, i + v_i), (255, 255, 0))
                cv2.arrowedLine(optical_flow, (j, i), (j + u_i, i + v_i), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow('Optical flow Horn&Schunck', result)
    cv2.waitKey(50)

    return result