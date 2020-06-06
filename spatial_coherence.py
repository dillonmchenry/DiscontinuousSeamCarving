import cv2
import imageio
import numpy as np


def calculate_transition_cost(xa, x, row, rowAbove):
    sum = 0
    for j in range(xa, x):
        gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
        if j < x:
            sum += gradient
        if j > xa:
            sum += gradient

    return sum


def compute_spatial_coherence_cost(frame, window):
    height, width = frame.shape[:2]
    cost_map = np.zeros((height, width))
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_scale = np.asarray(gray_scale, dtype=int)

    for y in range(0, height):
        gradients = []
        for x in range(0, width):
            row = gray_scale[y]
            rowAbove = gray_scale[y - 1]

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

            left_bound = window
            right_bound = window

            # checks left boundary
            if x < window:
                left_bound -= window - x
            # checks right boundary
            elif x + window > len(row) - 1:
                right_bound -= x + window - len(row)

            if not gradients:
                # calculates and sums transition for all pixels in window radius
                for i in range(-left_bound, right_bound + 1):
                    xa = x + i
                    gradients.append(calculate_transition_cost(xa, x, row, rowAbove))

            else:
                # Checks for boundary cases
                if left_bound >= window:
                    gradients.pop()
                if right_bound < window:
                    gradients.append(calculate_transition_cost(right_bound, x, row, rowAbove))

            verticalCost = sum(gradients)

            cost_map[y][x] = verticalCost + horizontalCost

    return cost_map


if __name__ == '__main__':
    im = imageio.imread('lawn_mower.jpg')
    spatial_map = compute_spatial_coherence_cost(im, 10)
    spatial_map = spatial_map / np.max(spatial_map) * 255
    cv2.imwrite("spatial_demo.jpg", spatial_map)
