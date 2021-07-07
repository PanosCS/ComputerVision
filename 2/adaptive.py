import numpy as np
from PIL import Image
from math import floor
import time
import sys


def grayscaling(A):
    if len(A.shape) == 3:
        return np.mean(A, axis=2)
    return A


def get_neighbors(R, i, j, d):
    return R[max(i - d, 0): min(i + d + 1, R.shape[0]), max(j - d, 0):min(j + d + 1, R.shape[1])].flatten()


def calc(W, k):
    pixels1 = W[W <= k]
    pixels2 = W[W > k]

    mu1 = np.mean(pixels1)
    mu2 = np.mean(pixels2)
    mu_all = np.mean(W)

    pi1 = len(pixels1) / (len(W))
    pi2 = len(pixels2) / (len(W))
    antikeimeniki_synartisi = pi1 * (mu1 - mu_all) ** 2 + pi2 * (mu2 - mu_all) ** 2
    return (antikeimeniki_synartisi)


def otsu(image, window_size):
    d = floor(window_size / 2)  # width size for neighbors function.
    # 2d array of (best) threshold values for every pixel in image
    ret = np.zeros(image.shape)
    row_size = image.shape[0]
    col_size = image.shape[1]

    for row in range(row_size):
        for col in range(col_size):
            k = 0
            best_s = 0
            # Returns a flattened array of neighbor pixels
            neighbors = get_neighbors(image, row, col, d)

            for i in range(1, 256):
                obj_otsu = calc(neighbors, i)

                if obj_otsu > best_s:
                    k = i
                    best_s = obj_otsu

            ret[row, col] = k

    return ret


def main():
    if len(sys.argv) != 4:
        print("False command given")
        print("python <script.py> <input_filename> <output_filename> <window_size>")
        sys.exit()

    input_filename = sys.argv[1]
    if sys.argv[2].endswith('.png'):
        output_filename = sys.argv[2]
    else:
        output_filename = sys.argv[2] + '.png'
    window_size = int(sys.argv[3])

    # Get image and Grayscale it
    R = np.array(Image.open(input_filename))
    R = grayscaling(R)

    thresholds = otsu(R, window_size)

    for row in range(R.shape[0]):
        for col in range(R.shape[1]):
            if R[row, col] <= thresholds[row, col]:
                R[row, col] = 0
            else:
                R[row, col] = 255

    Image.fromarray(R.astype(np.uint8)).save(output_filename)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Time elapsed: {} minutes'.format((end - start) / 60))
