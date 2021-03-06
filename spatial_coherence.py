import cv2
import imageio
import numpy as np
from numba import jit
import disc_video_carving


def calculate_transition_cost(xa, x, row, rowAbove):
    sum = 0
    for j in range(xa, x):
        if j < x:  # left side
            sum += abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
        if j > xa:  # right side
            sum += abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j - 1]))


def compute_spatial_coherence_cost_pixel(row, rowAbove, x, y, window):
    horizontalCost = 0
    verticalCost = 0

    # Border Cases
    if x == 0:
        horizontalCost = abs(abs(row[x] - row[x + 1]) - abs(row[x + 1] - row[x + 2]))
    elif x == row.shape[0] - 1:
        horizontalCost = abs(abs(row[x] - row[x - 1]) - abs(row[x - 1] - row[x - 2]))
    # Internal Pixel
    else:
        horizontalCost = abs(row[x - 1] - row[x]) + abs(row[x] - row[x + 1]) - abs(
            row[x - 1] - row[x + 1])

    # ----------------------Vertical Cost---------------------------------
    sum1 = 0
    sum2 = 0

    # Moves to the left (if possible)
    xa = x - window
    if xa < 0:
        xa = 0
    for j in range(xa, x):
        gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
        if j < x:
            sum1 += gradient
        if j > xa:
            sum2 += gradient

    # Moves to the right (if possible)
    xa = x + window
    if xa >= row.shape[0]:
        xa = row.shape[0]
    for j in range(x, xa):
        gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j - 1]))
        if j > x:
            sum1 += gradient
        if j < xa:
            sum2 += gradient

    verticalCost = sum1 + sum2

    return horizontalCost + verticalCost


@jit(nopython=True)
def compute_spatial_coherence_cost(frame, window):
    # Precompute vertical gradients
    height, width = frame.shape[:2]
    costMap = np.zeros((height, width))

    verticalGradients = np.zeros((height, width))
    DRGradients = np.zeros((height, width - 1))
    for i in range(0, frame.shape[0]):
        row = frame[i]
        if i > 0:
            rowAbove = frame[i - 1]
            verticalGradients[i] = np.absolute(row - rowAbove)
            DRGradients[i] = np.absolute(row[:-1] - rowAbove[1:])
        for j in range(0, frame.shape[1]):
            if j == 0:
                horizontalCost = np.absolute(np.absolute(row[j] - row[j + 1]) - np.absolute(row[j + 1] - row[j + 2]))
            elif j == row.shape[0] - 1:
                horizontalCost = np.absolute(np.absolute(row[j] - row[j - 1]) - np.absolute(row[j - 1] - row[j - 2]))
            # Internal Pixel
            else:
                horizontalCost = np.absolute(row[j - 1] - row[j]) + np.absolute(row[j] - row[j + 1]) - np.absolute(
                    row[j - 1] - row[j + 1])
            if i == 0:
                costMap[i][j] = horizontalCost
                continue
            leftBound = j - window if (j - window > 0) else 0
            rightBound = j + window if (j + window < row.shape[0]) else row.shape[0]
            verticalCost = np.sum(verticalGradients[i - 1][leftBound:rightBound]) + np.sum(
                DRGradients[i - 1][leftBound:rightBound])
            costMap[i][j] = verticalCost + horizontalCost
    return costMap


if __name__ == '__main__':
    video = disc_video_carving.read_video("bowser_dunk_480.m4v")
    frame = cv2.cvtColor(video[120], cv2.COLOR_BGR2GRAY)
    frame = np.asarray(frame, dtype=int)
    spatial_map2 = compute_spatial_coherence_cost(frame, 10)
    spatial_map2 = spatial_map2 / np.max(spatial_map2) * 255
    cv2.imwrite("bowser_spatial_demo.jpg", spatial_map2)

    im = imageio.imread('lawn_mower.jpg')
    spatial_map = compute_spatial_coherence_cost(im, 10)
    spatial_map = spatial_map / np.max(spatial_map) * 255
    cv2.imwrite("spatial_demo.jpg", spatial_map)
