import numpy as np
import numpy.linalg as nlin
import cv2
from numba import jit


def getDerivates(img1, img2, k_gauss):
    '''

    :param img1:
    :param img2:
    :param k_gauss:
    :return:
    '''

    # Gaussian filter to blur the images and improve temporal derive
    img1_gauss = cv2.GaussianBlur(img1, k_gauss, 0)
    img2_gauss = cv2.GaussianBlur(img2, k_gauss, 0)

    # IMG1
    dx_1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, 3)
    dy_1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, 3)

    # IMG2
    dx_2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, 3)
    dy_2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, 3)

    # Get the mean of the derives and the temporal derive
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


@jit
def computeOF_LK_pinv(fr1, fr2, window, step, A, B, k_gauss):
    '''

    :param fr1:
    :param fr2:
    :param window:
    :param A:
    :param B:
    :param k_gauss:
    :return:
    '''
    img1 = fr1 * 1/255
    img2 = fr2 * 1 / 255
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]

    # step = 1

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivates(img1, img2, k_gauss)
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

            # Initialization of matrix A
            A[0, 0] = Ix2
            A[1, 0] = A[0, 1] = Ixy
            A[1, 1] = Iy2

            pinv_A = np.matrix.round(nlin.pinv(A))

            # Initialization of matrix B
            B[0, 0] = Ixt
            B[1, 0] = Iyt

            # Set uv_desp vector
            uv_desp = np.matrix.round(np.dot(pinv_A, B))

            u = uv_desp[0]
            v = uv_desp[1]

            cv2.arrowedLine(img2, (jj, ii), (jj + u, ii + v), (255, 255, 0))
            cv2.arrowedLine(optical_flow, (jj, ii), (jj + u, ii + v), (255, 255, 0))

    result = np.concatenate((img2, optical_flow), axis= 1)
    cv2.imshow("Optical flow", result)
    cv2.waitKey(50)
    # cv2.destroyAllWindows()

    return result


@jit
def computeOF_LK_unrolled(fr1, fr2, window, step, k_gauss):

    img1 = fr1 * 1/255
    img2 = fr2 * 1 / 255
    optical_flow = np.zeros_like(img1)
    h, w = img1.shape[:2]

    # step = 1

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivates(img1, img2, k_gauss)
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

    # step = 1

    # Get derivatives and all the algorithm's matrix
    dx, dy, dt = getDerivates(img1, img2, k_gauss)
    Ix_2, Iy_2, Ixy_2, Ixt_2, Iyt_2 = multiplyMatrix(dx, dy, dt)

    h_lim = int(np.ceil(h / window - 1))
    w_lim = int(np.ceil(w / window - 1))