import cv2
import numpy as np
import argparse
import imageio as img

# ------------Things to Implement-------------------

# TODO: Determine spatial coherence of removing a pixel
# TODO: Determine temporal coherence cost of removing a pixel
# TODO: Determine saliency of removing a pixel
# TODO: Combine the Sc, Tc, and S into a weighted ratio M
# TODO: Implement image seam carving algorithm from
#  (http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html) to minimize m

# OF COURSE THERE'S OTHER STUFF I JUST CANT THINK OF IT RN

# -----------Captures Video Input------------------
def read_video(name):
    cap = cv2.VideoCapture(name)

    # Check if successfully
    if (cap.isOpened() == False):
        print("Error opening video file at \'" + name + "\'")

    video = []
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    # 4-D np array: frames, height, width, rgb
    return video


def write_video(video, name):
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m','p','4','v'), 30.0, (1920, 1080))

    for frame in video:
        # write the flipped frame
        out.write(np.array(frame))

    # Release everything if job is finished
    out.release()


def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def visualize(im, boolmask=None, rotate=False):
    SEAM_COLOR = np.array([255, 200, 200])
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis


def saliency_map(frame):
    height, width = frame.shape[:2]
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    energy = np.zeros((height, width))
    m = np.zeros((height, width))

    U = np.roll(gray_scale, 1, axis=0)
    L = np.roll(gray_scale, 1, axis=1)
    R = np.roll(gray_scale, -1, axis=1)

    cU = np.abs(R-L)
    cL = np.abs(U-L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, height):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    vis = visualize(energy)
    cv2.imwrite("forward_energy_demo.jpg", vis)


# currentFrame = numpy array
# previousSeam = numpy array of pairs of points
# x, y = point to compute in the current frame
def compute_temporal_coherence_cost_pixel(currentFrame, previousSeam, x, y):
    seamX = previousSeam[y][0]
    cost = 0
    for i in range(min(x, seamX), max(x, seamX)):
        channelSum1 = 0
        channelSum2 = 0
        for j in range(0, currentFrame.shape[2]):
            channelSum1 += currentFrame[y][i][j]
            channelSum2 += currentFrame[y][i+1][j]
        cost += abs(channelSum1 - channelSum2)
    return cost

def compute_temporal_coherence_cost(currentFrame, previousSeam):
    cost = []
    for i in range(0, currentFrame.shape[0]):
        cost.append([])
        for j in range(0, currentFrame.shape[1]):
            cost[i].append(compute_temporal_coherence_cost_pixel(currentFrame, previousSeam, j, i))
    return cost

def compute_spatial_coherence_cost_pixel(row, rowAbove, x, y):
    # If border pixel
    horizontalCost = 0
    if x == 0:
        horizontalCost = abs(abs(row[y][x] - row[y][x+1]) - abs(row[y][x+1] - row[y][x+2]))
    elif x == row.shape[0]:
        horizontalCost = abs(abs(row[y][x] - row[y][x-1]) - abs(row[y][x-1] - row[y][x-2]))
    else:
        horizontalCost = abs(row[y][x-1]-row[y][x]) + abs(row[y][x]-row[y][x+1]) - abs(row[y][x-1]-row[y][x+1])

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retargets a video to specified size")
    parser.add_argument('--video', type=str, help='The path to the video to retarget')
    parser.add_argument('--width', type=int, help='Width to retarget video to')
    parser.add_argument('--height', type=int, help='Height to retarget video to')
    parser.add_argument('--out', type=str, help='The path to store the output to')
    args = parser.parse_args()

    video = read_video(args.video)
    saliency_map(video[0])

    write_video(video, args.out)