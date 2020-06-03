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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retargets a video to specified size")
    parser.add_argument('--video', type=str, help='The path to the video to retarget')
    parser.add_argument('--width', type=int, help='Width to retarget video to')
    parser.add_argument('--height', type=int, help='Height to retarget video to')
    parser.add_argument('--out', type=str, help='The path to store the output to')
    args = parser.parse_args()

    # -----------Captures Video Input------------------

    cap = cv2.VideoCapture("bowser_dunk.mp4")
    out = cv2.VideoWriter("bowza.mp4", -1, 30.0, (1920, 1080))

    # Check if successfully
    if (cap.isOpened() == False):
        print("Error opening video file at \'" + str.args.video + "\'")

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
    video = np.array(video)

    # ----------Writes and Saves video file------------



    for frame in video:
        # write the flipped frame
        out.write(frame)

    # Release everything if job is finished
    out.release()


def saliency_map(img):
