import argparse
from myPackage import tools as tl
from myPackage import opticalFlow as of
from os.path import basename, altsep, exists
import cv2
from PIL import Image
import numpy as np
import time
from math import floor

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--frames",
                    help="-f Original frames path")
    ap.add_argument("-r", "--results",
                    help="-r Results path")
    args = vars(ap.parse_args())

    # Get folders with the sequences
    sequences = tl.natSort(tl.getSequences(args["frames"]))
    # print(sequences)

    # Configuration
    # mode: 0: LK-pinv, 1: LK-unrolled, 2: Horn&Schunck
    mode = 0
    save = False
    # 0: blocks, 1: grid, 2: sphere
    seq_idx = 0
    conf_seq = {0: '.gif', 1: '.gif', 2: '.jpg', 3: '.ppm'}
    ext = conf_seq[seq_idx]
    window = [6, 9, 12]
    n = 5
    k_gauss = (n, n)



    # Get the frames of the sequence and make results folder
    seq_selected = sequences[seq_idx]
    name_sequence = basename(seq_selected)
    results_path = altsep.join((args["results"], name_sequence))

    if not exists(results_path):
        tl.makeDir(results_path)

    frames = tl.natSort(tl.getSamples(seq_selected, ext))
    # print(frames)

    for win in window:
        step = floor(win / 2)
        start = time.time()

        # Compute the optical flow using all frames of the sequence and measure times
        for fr_idx in range(len(frames)-1):
            if ext == '.gif':
                img1 = Image.open(frames[fr_idx])
                img1_rgb = img1.convert('RGB')
                img1_rgb = cv2.cvtColor(np.array(img1_rgb), cv2.COLOR_RGB2BGR)

                img2 = Image.open(frames[fr_idx + 1])
                img2_rgb = img2.convert('RGB')
                img2_rgb = cv2.cvtColor(np.array(img2_rgb), cv2.COLOR_RGB2BGR)

                if mode == 0:
                    A = np.zeros((2, 2))
                    B = np.zeros((2, 1))
                    result = of.computeOF_LK_pinv(img1_rgb, img2_rgb, win, step, A, B, k_gauss)
                elif mode == 1:
                    result = of.computeOF_LK_unrolled(img1_rgb, img2_rgb, win, step, k_gauss)
                elif mode == 2:
                    n_iter = 50     # 10 - 100
                    lam_pond = 3    # 0.1 - 60
                    result = of.computeOF_HS(img1_rgb, img2_rgb, win, step, k_gauss, n_iter, lam_pond)

                if save:
                    result_name = altsep.join((results_path, ''.join((name_sequence, '-', str(win), '-', str(fr_idx), '-', str(fr_idx + 1), '.png'))))
                    cv2.imwrite(result_name, result)

            elif ext == '.ppm' or ext == '.jpg':
                img1 = cv2.imread(frames[fr_idx])
                img2 = cv2.imread(frames[fr_idx + 1])

                if mode == 0:
                    result = of.computeOF_LK_pinv(img1, img2, win, step, A, B, k_gauss)
                elif mode == 1:
                    result = of.computeOF_LK_unrolled(img1, img2, win, step, k_gauss)
                elif mode == 2:
                    n_iter = 50  # 10 - 100
                    lam_pond = 3  # 0.1 - 60
                    result = of.computeOF_HS(img1, img2, win, step, k_gauss, n_iter, lam_pond)
                if save:
                    result_name = altsep.join((results_path, ''.join((name_sequence, '-', str(win), '-', str(fr_idx), '-', str(fr_idx + 1), '.png'))))
                    cv2.imwrite(result_name, result)

        time_taken = time.time() - start
        print("Optical flow takes {:0.3f} seconds for sequence '{}' with window_size = {}".format(time_taken, name_sequence, win))

    cv2.destroyAllWindows()
    # exit(2)
