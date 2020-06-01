import cv2
import numpy as np

# ------------Things to Implement-------------------

# TODO: Determine spatial coherence of removing a pixel
# TODO: Determine temporal coherence cost of removing a pixel
# TODO: Determine saliency of removing a pixel
# TODO: Combine the Sc, Tc, and S into a weighted ratio M
# TODO: Implement image seam carving algorithm from
#  (http://www.faculty.idc.ac.il/arik/SCWeb/imret/index.html) to minimize m

# OF COURSE THERE'S OTHER STUFF I JUST CANT THINK OF IT RN


# -----------Captures Video Input------------------

cap = cv2.VideoCapture('bowser_dunk.mp4')

# Check if successfully
if (cap.isOpened() == False):
    print("Error opening video file")

video = []
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        video.append(frame)
    else:
        break
cap.release()
# 4-D np array: frames, height, width, rgb
video = np.array(video)

# ----------Writes and Saves video file------------

out = cv2.VideoWriter('bowza.mp4', -1, 30.0, (1920, 1080))

for frame in video:
    if True:
        # write the flipped frame
        out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
