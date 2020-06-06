import cv2
import imageio
import numpy as np
import argparse
import imageio as img
from numba import jit

import seam
import spatial_coherence

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
    #cv2.imshow("visualization", vis)
    #cv2.waitKey(1)
    return vis

def saliency_map(gray_scale):
    height, width = gray_scale.shape[:2]

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

    return energy.astype(np.uint8)

def carve_seams(frame):
    seams = [[] for i in range(frame.shape[1])]
    energies = np.zeros(frame.shape[1])

    min_energy = np.amin(frame[0, 0:2])
    y = np.where(frame[0, 0:2] == min_energy)[0][0] # calculates seam for top left corner
    seams[0].append([0, y]) 
    seams[0].append([1, 0])
    energies[0] += min_energy + frame[1][0] 

    for j in range(1, frame.shape[1] - 1):
        min_energy = np.amin(frame[0, j-1:j+2])
        y = j - 1 + np.where(frame[0, j-1:j+2] == min_energy)[0][0] #calculates initial seams for the second row, from the first row
        seams[j].append([0, y])
        seams[j].append([1, j]) 
        energies[j] += min_energy + frame[1][j] 

    min_energy = np.amin(frame[0, frame.shape[1]-2:frame.shape[1]])
    y = frame.shape[1] - 2 + np.where(frame[0, frame.shape[1]-2:frame.shape[1]] == min_energy)[0][0] #calculates seam from top left
    seams[frame.shape[1]-1].append([0, y])
    seams[frame.shape[1]-1].append([1, frame.shape[1]-1]) 
    energies[frame.shape[1]-1] += min_energy + frame[1][frame.shape[1]- 1] 
        
    for i in range(2, frame.shape[0]): #loops through all subsequent rows
        new_seams = [[] for i in range(frame.shape[1])]
        new_energies = np.zeros(frame.shape[1])

        min_energy = np.amin(energies[0:2])
        y = np.where(energies[0:2] == min_energy)[0][0] #calculates the new seam to the left
        new_seams[0] = seams[y].copy()
        new_seams[0].append([i, 0]) 
        new_energies[0] = min_energy + frame[i][0] 

        for j in range(1, frame.shape[1] - 1):
            min_energy = np.amin(energies[j-1:j+2])
            y = j - 1 + np.where(energies[j-1:j+2] == min_energy)[0][0] #calculates all middle seams
            new_seams[j] = seams[y].copy()
            new_seams[j].append([i, j])  
            new_energies[j] = min_energy + frame[i][j]

        min_energy = np.amin(energies[frame.shape[1]-2:frame.shape[1]])
        y = frame.shape[1] - 2 + np.where(energies[frame.shape[1]-2:frame.shape[1]] == min_energy)[0][0] #calculates last seam of a row
        new_seams[frame.shape[1]-1] = seams[y].copy()
        new_seams[frame.shape[1]-1].append([i, frame.shape[1]-1]) 
        new_energies[frame.shape[1]-1] = min_energy + frame[i][frame.shape[1]- 1] 

        seams = new_seams.copy()
        energies = new_energies[:]

    return (seams, energies)

def carve_seams_piecewise(frame, width):
    frame = np.array(frame)
    seams = [[] for i in range(frame.shape[1])]
    energies = np.zeros(frame.shape[1])

    for i in range(frame.shape[1]):
        seams[i].append([0, i])
        energies[i] = frame[0, i]

    for y in range(1, frame.shape[0]):
        new_seams = [[] for i in range(frame.shape[1])]
        new_energies = [0 for i in range(frame.shape[1])]
        for x in range(frame.shape[1]):
            leftBound = x - width if x > width else 0
            rightBound = x + width if x + width < frame.shape[1] else frame.shape[1] - 1

            min_energy = np.amin(energies[leftBound:rightBound])
            min_index = np.where(energies == min_energy)[0][0]
            new_seams[x] = seams[min_index].copy()
            new_seams[x].append([y, min_index])
            new_energies[x] = min_energy + frame[y, min_index]

        seams = new_seams.copy()
        energies = new_energies.copy()


    return (seams, energies)

def highlight_seam(frame, seam):
    new_frame = frame.copy()
    for pixel in seam:
        new_frame[pixel[0], pixel[1]] = [255, 180, 180]
    return new_frame

@jit(nopython=True)
def compute_temporal_coherence_cost(currentFrame, previousSeam):
    costMap = []
    for i in range(0, currentFrame.shape[0]):
        costMap.append([0 for x in range(currentFrame.shape[1])])
        cumulativeCost = 0
        for j in range(previousSeam[i][1]-1, -1, -1):
            # channels1 = np.linalg.norm(currentFrame[i][j])
            # channels2 = np.linalg.norm(currentFrame[i][j + 1])
            cumulativeCost += np.absolute(currentFrame[i][j] - currentFrame[i][j + 1])
            costMap[i][j] = cumulativeCost
        cumulativeCost = 0
        costMap[i][previousSeam[i][1]] = 0
        for j in range(previousSeam[i][1]+1, currentFrame.shape[1]):
            # channels1 = np.linalg.norm(currentFrame[i][j])
            # channels2 = np.linalg.norm(currentFrame[i][j - 1])
            cumulativeCost += np.absolute(currentFrame[i][j] - currentFrame[i][j - 1])
            costMap[i][j] = cumulativeCost
            
    return costMap

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

def retarget_video(video, width, height):
    widthDif = video.shape[2] - width
    heightDif = video.shape[1] - height
    # Shrink first
    if (widthDif > 0):
        pass
    if (heightDif > 0):
        pass

    # Then expand
    if (widthDif < 0):
        pass

    if (heightDif < 0):
        pass

    newVideo = []
    for i in range(0, abs(widthDif) + abs(heightDif)):
        for j in range(0, video.shape[0]):
            verticalSeams, verticalEnergies = carve_seams(video[j])
            minVerticalSeam, minVerticalEnergy = seam.get_n_seams(verticalSeams, verticalEnergies, 1)

            horizontalSeams, horizontalEnergies = seam.carve_seams(video[j].T)
            minHorizontalSeam, minHorizontalEnergy = seam.get_n_seams(horizontalSeams, horizontalEnergies, 1)
            if (minVerticalEnergy < minHorizontalEnergy):
                # Do vertical seam
                if (widthDif > 0):
                    # Shrinking horizontally
                    pass

                else:
                    # Expanding horizontally
                    pass
                # Remove vertical seam from frame by copying over elements from current frame
            else:
                if (heightDif > 0):
                    # Shrinking vertically
                    pass
                else:
                    # Expanding vertically
                    pass

            pass

# Weights is tuple of (SC:TC:S)
def getPixelMeasures(frameIn, spatialWindow, weights, previousSeam=None):
    frame = cv2.cvtColor(frameIn, cv2.COLOR_BGR2GRAY)
    frame = np.asarray(frame, dtype=int)
    spatialWeight, temporalWeight, saliencyWeight = weights / np.sum(weights)
    spatialMap = compute_spatial_coherence_cost(frame, spatialWindow)
    spatialMap = spatialMap / np.max(spatialMap) * spatialWeight
    saliency = saliency_map(frame)
    saliency = saliency / np.max(saliency) * saliencyWeight
    if (previousSeam is None):
        return (spatialMap + saliency) * 255
    temporalMap = compute_temporal_coherence_cost(frame, previousSeam)
    temporalMap = temporalMap / np.max(temporalMap) * temporalWeight
    return (spatialMap + saliency + temporalMap) * 255

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retargets a video to specified size")
    parser.add_argument('--video', type=str, help='The path to the video to retarget')
    parser.add_argument('--width', type=int, help='Width to retarget video to')
    parser.add_argument('--height', type=int, help='Height to retarget video to')
    parser.add_argument('--out', type=str, help='The path to store the output to')
    args = parser.parse_args()

    print("INFO: Reading Video: ", args.video)
    video = read_video(args.video)
    print("INFO: Finished Reading Video")

    print("INFO: Calculating Saliency Map")
    saliency_frame = saliency_map(video[120])

    print("INFO: Calculating Seams")
    seam, energies = carve_seams(saliency_frame)
    print("INFO: Finished Calculating Seams")
    min_index = np.where(energies == np.amin(energies))[0][::-1]
    min_index = min_index[0] 
    min_seam = seam[min_index]

    mask = highlight_seam(video[120], min_seam)
    print("INFO: Saving New Image")
    cv2.imwrite("saliency_seam_demo.jpg", mask)

    print("INFO: Calculating Temporal Cost to Next Frame")
    temporal_map = compute_temporal_coherence_cost(video[121], min_seam)
    temporal_map = temporal_map / np.max(temporal_map) * 255
    cv2.imwrite("temporal_demo.jpg", temporal_map)
    #print("INFO: Saving New Image")
    #cv2.imwrite("temporal_map_demo.jpg", temporal_map.astype(np.uint8))

    print("INFO: Calculating Merged Maps")
    costMap = getPixelMeasures(video[121], 15, (5, 1, 2), min_seam)
    cv2.imwrite("cost_map_demo.jpg", costMap)

    print("INFO: Calculating Seams from Temporal Cost")
    seam2, energies2 = carve_seams_piecewise(costMap, 3)
    min_index2 = np.where(energies2 == np.amin(energies2))[0][::-1]
    min_index2 = min_index2[0] 
    min_seam2 = seam2[min_index]

    mask2 = highlight_seam(video[121], min_seam2)
    print("INFO: Saving New Image")
    cv2.imwrite("cost_seam_demo.jpg", mask2)

    #write_video(video, args.out)
