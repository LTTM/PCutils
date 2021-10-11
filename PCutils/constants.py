import numpy as np

kitti_colors = np.zeros((20, 3))
kitti_colors[0] = [0, 0, 0]  # unlabeled
kitti_colors[1] = [245, 150, 100]  # car
kitti_colors[2] = [245, 230, 100]  # bike
kitti_colors[3] = [150, 60, 30]  # motorcycle
kitti_colors[4] = [180, 30, 80]  # truck
kitti_colors[5] = [255, 0, 0]  # other-vehicle
kitti_colors[6] = [30, 30, 255]  # person
kitti_colors[7] = [200, 40, 255]  # bicyclist
kitti_colors[8] = [90, 30, 150]  # motorcyclist
kitti_colors[9] = [255, 0, 255]  # road
kitti_colors[10] = [255, 150, 255]  # parking
kitti_colors[11] = [75, 0, 75]  # sidewalk
kitti_colors[12] = [75, 0, 175]  # other-ground
kitti_colors[13] = [0, 200, 255]  # building
kitti_colors[14] = [50, 120, 255]  # fence
kitti_colors[15] = [0, 175, 0]  # vegetation
kitti_colors[16] = [0, 60, 135]  # trunck
kitti_colors[17] = [80, 240, 150]  # terrain
kitti_colors[18] = [150, 240, 255]  # pole
kitti_colors[19] = [0, 0, 255]  # traffic sign
kitti_colors = kitti_colors / 255.0

