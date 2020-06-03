import cv2
import numpy as np
import argparse

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


# currentFrame = numpy array
# previousSeam = numpy array of pairs of points
# x, y = point to compute in the current frame
def compute_temporal_coherence_cost(currentFrame, previousSeam, x, y):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retargets a video to specified size")
    parser.add_argument('--video', type=str, help='The path to the video to retarget')
    parser.add_argument('--width', type=int, help='Width to retarget video to')
    parser.add_argument('--height', type=int, help='Height to retarget video to')
    parser.add_argument('--out', type=str, help='The path to store the output to')
    args = parser.parse_args()

    video = read_video(args.video)

    write_video(video, args.out)