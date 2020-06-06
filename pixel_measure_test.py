import disc_video_carving
import seam
import cv2
import numpy as np


if __name__ == '__main__':
    video = disc_video_carving.read_video("bowser_dunk_480.m4v")
    costMap1 = disc_video_carving.getPixelMeasures(video[120], 10, (5, 1, 2))
    seam, energies = disc_video_carving.carve_seams(costMap1)
    min_index = np.where(energies == np.amin(energies))[0][::-1]
    min_index = min_index[0]
    min_seam = np.array(seam[min_index])
    costMap2 = disc_video_carving.getPixelMeasures(video[121], 10, (5, 1, 2), min_seam)
    seam, energies = disc_video_carving.carve_seams(costMap2)
    min_index = np.where(energies == np.amin(energies))[0][::-1]
    min_index = min_index[0]
    min_seam2 = np.array(seam[min_index])
    mask1 = disc_video_carving.highlight_seam(video[120], min_seam)
    mask2 = disc_video_carving.highlight_seam(video[121], min_seam2)
    cv2.imwrite("costmap1_demo.jpg", costMap1)
    cv2.imwrite("costmap2_demo.jpg", costMap2)
    cv2.imwrite("seam1_demo.jpg", mask1)
    cv2.imwrite("seam2_demo.jpg", mask2)


