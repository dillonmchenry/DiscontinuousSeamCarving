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
    if (xa < 0):
        xa = 0
    for j in range(xa, x):
        gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j + 1]))
        if j < x:
            sum1 += gradient
        if j > xa:
            sum2 += gradient

    # Moves to the right (if possible)
    xa = x + window
    if (xa >= row.shape[0]):
        xa = row.shape[0]
    for j in range(x, xa):
        gradient = abs(abs(row[j] - rowAbove[j]) - abs(row[j] - rowAbove[j - 1]))
        if j > x:
            sum1 += gradient
        if j < xa:
            sum2 += gradient

    verticalCost = sum1 + sum2

    return horizontalCost + verticalCost

def compute_spatial_coherence_cost2(frame, window):
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


def compute_spatial_coherence_cost(frameIn, window):
    #Precompute vertical gradients
    height, width = frameIn.shape[:2]
    costMap = np.zeros((height, width))
    frame = cv2.cvtColor(frameIn, cv2.COLOR_BGR2GRAY)
    frame = np.asarray(frame, dtype=int)
    verticalGradients = np.zeros((height, width))
    DRGradients = np.zeros((height, width-1))
    for i in range(0, frame.shape[0]):
        row = frame[i]
        if (i > 0):
            rowAbove = frame[i - 1]
            verticalGradients[i] = abs(row - rowAbove)
            DRGradients[i] = abs(row[:-1] - rowAbove[1:])
        for j in range(0, frame.shape[1]):
            if j == 0:
                horizontalCost = abs(abs(row[j] - row[j + 1]) - abs(row[j + 1] - row[j + 2]))
            elif j == row.shape[0] - 1:
                horizontalCost = abs(abs(row[j] - row[j - 1]) - abs(row[j - 1] - row[j - 2]))
            # Internal Pixel
            else:
                horizontalCost = abs(row[j - 1] - row[j]) + abs(row[j] - row[j + 1]) - abs(
                    row[j - 1] - row[j + 1])
            if (i == 0):
                costMap[i][j] = horizontalCost
                continue
            leftBound = j - window if (j - window > 0) else 0
            rightBound = j + window if (j + window < row.shape[0]) else row.shape[0]
            verticalCost = np.sum(verticalGradients[i-1][leftBound:rightBound]) + np.sum(DRGradients[i-1][leftBound:rightBound])
            costMap[i][j] = verticalCost + horizontalCost
    return costMap

if __name__ == '__main__':
    video = disc_video_carving.read_video("bowser_dunk_480.m4v")
    spatial_map2 = compute_spatial_coherence_cost2(video[120], 10)
    spatial_map2 = spatial_map2 / np.max(spatial_map2) * 255
    cv2.imwrite("bowser_spatial_demo.jpg", spatial_map2)

    im = imageio.imread('lawn_mower.jpg')
    spatial_map = compute_spatial_coherence_cost(im, 10)
    spatial_map = spatial_map / np.max(spatial_map) * 255
    cv2.imwrite("bowser_spatial_demo.jpg", spatial_map)
