import disc_video_carving
import cv2
import numpy as np


if __name__ == '__main__':
    video = disc_video_carving.read_video("bowser_dunk_480.m4v")
    print("FRAME 1")
    costMap1 = disc_video_carving.getPixelMeasures(video[120], 10, (5, 1, 2))
    print("Carving seams")
    seam, energies = disc_video_carving.carve_seams_piecewise(costMap1, 5)
    min_index = np.where(energies[-1] == np.amin(energies[-1]))[0]
    min_index = min_index[0]
    min_seam = np.array(disc_video_carving.get_seam(seam, min_index))
    print("FRAME 2")
    costMap2 = disc_video_carving.getPixelMeasures(video[121], 10, (5, 1, 2), min_seam)
    print("Carving seams")
    seam, energies = disc_video_carving.carve_seams_piecewise(costMap2, 5)
    min_index = np.where(energies[1] == np.amin(energies[1]))[0]
    min_index = min_index[0]
    min_seam2 = np.array(disc_video_carving.get_seam(seam, min_index))
    new_frame = disc_video_carving.remove_seam(video[121], seam, min_index)
    print("SAVING IMAGES")
    mask1 = disc_video_carving.highlight_seam(video[120], min_seam)
    mask2 = disc_video_carving.highlight_seam(video[121], min_seam2)
    cv2.imwrite("costmap1_demo.jpg", costMap1)
    cv2.imwrite("costmap2_demo.jpg", costMap2)
    cv2.imwrite("seam1_demo.jpg", mask1)
    cv2.imwrite("seam2_demo.jpg", mask2)
    cv2.imwrite("removed_seam_demo.jpg", new_frame)


