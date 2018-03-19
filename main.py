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
    # print(sequences)

    # Configuration
    # mode: 0: LK-pinv, 1: LK-unrolled, 2: Horn&Schunck
    mode_idx = 2
    mode_names = {0: "Lucas-Kanade (pinv)", 1: "Lucas-Kanade (unrolled)", 2: "Horn&Schunck"}
    save = False

    # 0: blocks, 1: grid, 2: postit, 3: sphere
    seq_idx = 0
    conf_seq = {0: '.gif', 1: '.gif', 2: '.jpg', 3: '.ppm'}

    # For Horn&Schonck algorithm
    n_iter = 50  # 10 - 100
    lam_pond = 1  # 0.1 - 60

    ext = conf_seq[seq_idx]
    window = [5, 9, 15]
    n = 5
    k_gauss = (n, n)
    plt_step = 3

    # Get the frames of the sequence and make results folder
    seq_selected = sequences[seq_idx]
    name_sequence = basename(seq_selected)
    results_path = altsep.join((args["results"], name_sequence))

    if not exists(results_path) and save:
        tl.makeDir(results_path)

    frames = tl.natSort(tl.getSamples(seq_selected, ext))
    # print(frames)

    for win in window:
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

                if mode_idx == 0:
                    result = of.computeOF_LK_pinv(img1_rgb, img2_rgb, win, k_gauss, plt_step)
                elif mode_idx == 1:
                    result = of.computeOF_LK_unrolled(img1_rgb, img2_rgb, win, k_gauss, plt_step)
                elif mode_idx == 2:
                    result = of.computeOF_HS(img1_rgb, img2_rgb, win, k_gauss, n_iter, lam_pond, plt_step)

                if save:
                    if mode_idx != 2:
                        result_name = altsep.join((results_path, ''.join((str(mode_idx), '-', name_sequence, '-', str(win), '-', str(fr_idx), '-', str(fr_idx + 1), '.png'))))
                    else:
                        result_name = altsep.join((results_path, ''.join((str(mode_idx), '-', n_iter, '-', lam_pond, '-', name_sequence, '-',
                                                                          str(win), '-', str(fr_idx), '-',
                                                                          str(fr_idx + 1), '.png'))))
                    cv2.imwrite(result_name, result)

            elif ext == '.ppm' or ext == '.jpg':
                img1 = cv2.imread(frames[fr_idx])
                img2 = cv2.imread(frames[fr_idx + 1])

                if mode_idx == 0:
                    result = of.computeOF_LK_pinv(img1, img2, win, k_gauss, plt_step)
                elif mode_idx == 1:
                    result = of.computeOF_LK_unrolled(img1, img2, win, k_gauss, plt_step)
                elif mode_idx == 2:
                    result = of.computeOF_HS(img1, img2, win, k_gauss, n_iter, lam_pond, plt_step)

                if save:
                    if mode_idx != 2:
                        result_name = altsep.join((results_path, ''.join((str(mode_idx), '-', name_sequence, '-', str(win), '-', str(fr_idx), '-', str(fr_idx + 1), '.png'))))
                    else:
                        result_name = altsep.join((results_path, ''.join((str(mode_idx), '-', n_iter, '-', lam_pond, '-', name_sequence, '-',
                                                                          str(win), '-', str(fr_idx), '-',
                                                                          str(fr_idx + 1), '.png'))))
                    cv2.imwrite(result_name, result)

        time_taken = time.time() - start
        print("Optical flow ({}) takes {:0.3f} seconds for sequence '{}' with window_size = {}".format(mode_names[mode_idx], time_taken, name_sequence, win))

    print("\nExecution finished! \nExiting program...")
    cv2.destroyAllWindows()
    cv2.waitKey(100)
# exit(0)
