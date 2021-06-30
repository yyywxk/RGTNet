# -*- coding: utf-8 -*-

from skimage.feature import local_binary_pattern
import cv2
import os

# settings for LBP
radius = 8
# radius = 16
n_points = 8 * radius
METHOD = 'default'

from_path = './image/'  # original image path
to_path = './image1/'  # LBP output path
if not os.path.exists(to_path1):
    os.mkdir(to_path1)


img_list = os.listdir(from_path)
print('There are {} images\n Start produce LBP feature maps......\n'.format(len(img_list)))
for i, item in enumerate(img_list):
    img = cv2.imread(from_path + item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    cv2.imwrite(to_path1 + item, lbp)
    print('Processing {}/{}!'.format(i+1, len(img_list)))

print('Finished!')
