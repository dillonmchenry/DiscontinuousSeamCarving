import cv2
from numpy import np


# Sum of different in gradients both diagonal and vertical
def calc_sum(xa, xb, row, rowAbove):
    sum1 = 0
    for x in range(xa, xb):
        sum1 += abs(abs(row[x] - rowAbove[x]) - abs(row[x] - rowAbove[x + 1]))

    return sum1


def compute_spatial_coherence_cost_pixel(row, rowAbove, x, y, window):
    horizontalCost = 0
    verticalCost = 0

    # Border Cases
    if x == 0:
        horizontalCost = abs(abs(row[y][x] - row[y][x + 1]) - abs(row[y][x + 1] - row[y][x + 2]))
    elif x == row.shape[0]:
        horizontalCost = abs(abs(row[y][x] - row[y][x - 1]) - abs(row[y][x - 1] - row[y][x - 2]))
    # Internal Pixel
    else:
        horizontalCost = abs(row[y][x - 1] - row[y][x]) + abs(row[y][x] - row[y][x + 1]) - abs(
            row[y][x - 1] - row[y][x + 1])

    # ----------------------Vertical Cost---------------------------------
    sum1 = 0
    sum2 = 0

    for i in range(1, window):

        # Moves to the left (if possible)
        if x - i >= 0:
            xa = x - i
            sum1 += calc_sum(xa, x - 1, row, rowAbove)
            sum2 += calc_sum(xa + 1, x, row, rowAbove)

        # Moves to the right (if possible)
        if x + i <= len(row) - 1:
            xa = x + i
            sum1 += calc_sum(x, xa - 1, row, rowAbove)
            sum2 += calc_sum(x + 1, xa, row, rowAbove)

    return sum1 + sum2


def compute_spatial_coherence_cost(frame, window):
    height, width = frame.shape[:2]
    cost_map = np.zeros((height, width))
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, height):
        for j in range(0, width):
            cost_map[i][j] = compute_spatial_coherence_cost_pixel(frame[i], frame[i - 1], j, i, window)

    return cost_map


if __name__ == '__main__':
    print("hi")
