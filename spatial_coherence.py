import cv2
import imageio
import numpy as np


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

    for i in range(1, window):

        # Moves to the left (if possible)
        if x - i >= 0:
            xa = x - i
            for j in range(xa, x):
                gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
                if j < x:
                    sum1 += gradient
                if j > xa:
                    sum2 += gradient

        # Moves to the right (if possible)
        if x + i <= len(row) - 1:
            xa = x + i
            for j in range(xa, x):
                gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
                if j < x:
                    sum1 += gradient
                if j > xa:
                    sum2 += gradient

    verticalCost = sum1 + sum2

    return horizontalCost + verticalCost


def compute_spatial_coherence_cost(frame, window):
    height, width = frame.shape[:2]
    cost_map = np.zeros((height, width))
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_scale = np.asarray(gray_scale, dtype=int)

    for i in range(0, height):
        for j in range(0, width):
            cost_map[i][j] = compute_spatial_coherence_cost_pixel(gray_scale[i], gray_scale[i - 1], j, i, window)

    return cost_map


if __name__ == '__main__':
    im = imageio.imread('lawn_mower.jpg')
    spatial_map = compute_spatial_coherence_cost(im, 10)
    spatial_map = spatial_map / np.max(spatial_map) * 255
    cv2.imwrite("spatial_demo.jpg", spatial_map)

