import cv2
import imageio
import numpy as np
import disc_video_carving


def calculate_transition_cost(xa, x, row, rowAbove):
    sum = 0
    for j in range(xa, x):
        if j < x: # left side
            sum += abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
        if j > xa: # right side
            sum += abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j - 1]))

    return sum


def compute_spatial_coherence_cost3(frame, window):
    height, width = frame.shape[:2]
    cost_map = np.zeros((height, width))
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_scale = np.asarray(gray_scale, dtype=int)

    for y in range(0, height):
        gradientsV = []
        gradientsDR = []
        gradientsDL = []
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
                horizontalCost = abs(row[x - 1] - row[x]) + abs(row[x] - row[x + 1]) - abs(row[x - 1] - row[x + 1])

            # ----------------------Vertical Cost---------------------------------
            if y > 0:
                if len(gradientsV) == 0:
                    # calculates and sums transition for all pixels in window radius
                    for i in range(window):
                        gradientsV.append(abs(row[i] - rowAbove[i]))
                        gradientsDL.append(abs(row[i + 1] - rowAbove[i]))
                    gradientsV.append(abs(row[window] - rowAbove[window]))

                leftBound = x if x < window else window
                rightBound = window + leftBound if x + window < width else window + leftBound + (width - (x + window))
                verticalCost = abs(sum(gradientsV[0:leftBound]) - sum(gradientsDR)) * 2 + abs(sum(gradientsV[leftBound:rightBound]) - sum(gradientsDL)) * 2
                if x != 0:
                    verticalCost -= gradientsV[0]
                if x != width - 1:
                    verticalCost -= gradientsV[len(gradientsV) - 1]

                verticalCost = abs(verticalCost)

                if x < width - 1:
                    gradientsDR.append(abs(row[x] - rowAbove[x + 1]))
                    gradientsDL.pop(0)
                if x > window:
                    gradientsV.pop(0)
                    gradientsDR.pop(0)
                if x + window < width:
                    gradientsV.append(abs(row[x + window] - rowAbove[x + window]))
                if x + window < width - 1:
                    gradientsDL.append(abs(row[x + window + 1] - rowAbove[x + window]))

            cost_map[y][x] = verticalCost + horizontalCost

    return cost_map


if __name__ == '__main__':
    # im = imageio.imread('lawn_mower.jpg')
    # spatial_map = compute_spatial_coherence_cost2(im, 10)
    # spatial_map = spatial_map / np.max(spatial_map) * 255
    # cv2.imwrite("spatial_demo.jpg", spatial_map)

    video = disc_video_carving.read_video("bowser_dunk_480.m4v")
    spatial_map = compute_spatial_coherence_cost3(video[120], 10)
    spatial_map = spatial_map / np.max(spatial_map) * 255
    cv2.imwrite("bowser_spatial_demo.jpg", spatial_map)
