import argparse
from myPackage import tools as tl
from myPackage import opticalFlow as of
from os.path import basename, altsep, exists
import cv2
from PIL import Image
import numpy as np
import time

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--frames",
                    help="-f Original frames path")
    ap.add_argument("-r", "--results",
                    help="-r Results path")
    args = vars(ap.parse_args())

    # Get folders with the sequences
    sequences = tl.natSort(tl.getSequences(args["frames"]))
    print(sequences)

    # Configuration
    color = True
    # 0: blocks, 1: grid, 2: sphere
    seq_idx = 2
    conf_seq = {0: '.gif', 1: '.gif', 2: '.jpg', 3: 'ppm'}
    ext = conf_seq[seq_idx]
    window = 13
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    uv_desp = np.zeros((2, 1))
    n = 5
    k_gauss = (n, n)



    # Get the frames of the sequence and make results folder
    seq_selected = sequences[seq_idx]
    name_sequence = basename(seq_selected)
    results_path = altsep.join((args["results"], name_sequence))

    if not exists(results_path):
        tl.makeDir(results_path)

    frames = tl.natSort(tl.getSamples(seq_selected, ext))

    start = time.time()
    # Compute the optical flow using all frames of the sequence and measure times
    for fr_idx in range(len(frames)-1):
        # print(frame, frame+1)
        if ext == '.gif':
            img1 = Image.open(frames[fr_idx])
            img1_rgb = img1.convert('RGB')
            img1_rgb = cv2.cvtColor(np.array(img1_rgb), cv2.COLOR_RGB2BGR)

            img2 = Image.open(frames[fr_idx + 1])
            img2_rgb = img2.convert('RGB')
            img2_rgb = cv2.cvtColor(np.array(img2_rgb), cv2.COLOR_RGB2BGR)

            result = of.computeOF(img1_rgb, img2_rgb, window, A, B, uv_desp, k_gauss)
            result_name = tl.join(name_sequence, '-', str(window), '-', str(fr_idx), '-', str(fr_idx + 1), '.png')
            # cv2.imwrite(result_name, result)

        elif ext == '.ppm' or ext == '.jpg':
            img1 = cv2.imread(frames[fr_idx])
            img2 = cv2.imread(frames[fr_idx + 1])

            result = of.computeOF(img1, img2, window, A, B, uv_desp, k_gauss)
            result_name = tl.join(name_sequence, '-', str(window), '-', str(fr_idx), '-', str(fr_idx + 1), '.png')
            # cv2.imwrite(result_name, result)

    time_taken = time.time() - start
    print("Optical flow takes {:0.3f} seconds for sequence '{}' with window_size = {}".format(time_taken, name_sequence, window))

    exit(0)
