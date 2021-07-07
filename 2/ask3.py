import cv2
import numpy as np
import math
from sys import argv, exit


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Clicked on pixel: {},{}'.format(x, y))
        refPt.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", img)


# Calculate the distance between points using Euclidean distance
def euclidean_distance(C, X):
    matrix = []
    for i in range(X.shape[0]):
        min_value = math.inf
        for j in range(C.shape[0]):
            distance = (np.dot(X[i] - C[j], X[i] - C[j])) ** .5
            min_value = min(min_value, distance)
        matrix.append(min_value)
    return np.array(matrix)


def clockwise_order(pts):
    pts = np.array(pts)

    # sort the points based on  x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points
    leftMost = x_sorted[:2, :]
    rightMost = x_sorted[2:, :]

    # sort the left-most coordinates based on y-coordinates
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    # Top_left, and bottom_right
    (tl, bl) = leftMost

    # Use top_left as an anchor, an calculate the Euclidean distance between the Top_left and RightMost
    D = euclidean_distance(tl[np.newaxis], rightMost)
    #     print(D)
    # Bottom_right, and top_right
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates
    return np.array([tl, tr, br, bl], dtype="float32")


def transform(image, pts):
    # Order pts clockwise
    rect = clockwise_order(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # compute destination Points based on formula
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = perspective_transform_matrix(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def perspective_transform_matrix(src, dst):
    matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(dst).reshape(8)

    #     res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    res = np.linalg.solve(A, B)

    # Don't forget to add the last value of projection matrix (is always 1)
    return np.append(np.array(res).reshape(8), 1).reshape((3, 3))


def main():
    global refPt, img

    if len(argv) != 3:
        print('False arguments')
        exit()

    input_filename = argv[1]
    output_filename = argv[2]

    refPt = []
    img = cv2.imread(input_filename)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)

    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(refPt) != 4:
        raise ValueError("There must be 4 source points, found: {}".format(len(refPt)))

    warped = transform(img, refPt)

    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.imwrite(output_filename, warped)


if __name__ == "__main__":
    main()
