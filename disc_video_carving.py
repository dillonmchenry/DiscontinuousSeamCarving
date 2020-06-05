import cv2
import numpy as np
import argparse
import imageio as img
import seam

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
    print("INFO: Finished calculating Saliency Map")
    cv2.imwrite("forward_energy_demo.jpg", vis)

    return vis


def carve_seams(frame):
    frame = np.array(frame)
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

def highlight_seam_mask(frame, seam):
    new_frame = frame.copy()
    for pixel in seam:
        new_frame[pixel[0], pixel[1]] = [255, 180, 180]
    return new_frame

# currentFrame = numpy array
# previousSeam = numpy array of pairs of points
# x, y = point to compute in the current frame
def compute_temporal_coherence_cost_pixel(currentFrame, previousSeam, x, y):
    seamX = previousSeam[y][1]
    cost = 0
    for i in range(min(x, seamX), max(x, seamX)):
        channels1 = np.linalg.norm(currentFrame[y][i])
        channels2 = np.linalg.norm(currentFrame[y][i+1])
        cost += abs(channels1 - channels2)
    return cost

def compute_temporal_coherence_cost(currentFrame, previousSeam):
    costMap = []
    for i in range(0, currentFrame.shape[0]):
        costMap.append([])
        for j in range(0, currentFrame.shape[1]):
            costMap[i].append(compute_temporal_coherence_cost_pixel(currentFrame, previousSeam, j, i))
    return costMap

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

    print("INFO: Reading Video: ", args.video)
    video = read_video(args.video)

    print("INFO: Finished Reading Video")
    print("INFO: Calculating Saliency Map")
    saliency_frame = saliency_map(video[120])

    print("INFO: Calculating Seams")
    seam, energies = carve_seams(saliency_frame)
    print("INFO: Finished Calculating Seams")
    min_index = np.where(energies == np.amin(energies))[0][::-1]
    print(min_index)
    min_index = min_index[0] 
    print(min_index)
    min_seam = seam[min_index]
    mask = highlight_seam_mask(video[120], min_seam)

    vis2 = visualize(mask)
    cv2.imwrite("seam_demo.jpg", vis2)

    #write_video(video, args.out)
