



def compute_spatial_coherence_cost_pixel(row, rowAbove, x, y):
    horizontalCost = 0
    verticalCost = 0

    #Border Cases
    if x == 0:
        horizontalCost = abs(abs(row[y][x] - row[y][x+1]) - abs(row[y][x+1] - row[y][x+2]))
    elif x == row.shape[0]:
        horizontalCost = abs(abs(row[y][x] - row[y][x-1]) - abs(row[y][x-1] - row[y][x-2]))
    #Internal Pixel
    else:
        horizontalCost = abs(row[y][x-1]-row[y][x]) + abs(row[y][x]-row[y][x+1]) - abs(row[y][x-1]-row[y][x+1])

    #----------------------Vertical Cost---------------------------------

    #TODO:



    return None


if __name__ == '__main__':


