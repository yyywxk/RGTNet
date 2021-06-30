import os
import numpy as np
from PIL import Image
 
filepath = './image/'  # dataset path
pathDir = os.listdir(filepath)
 
R_channel = np.zeros(len(pathDir))
G_channel = np.zeros(len(pathDir))
B_channel = np.zeros(len(pathDir))
print('start !')
num = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    # img = imread(os.path.join(filepath, filename)) / 255.0
    img1 = Image.open(os.path.join(filepath, filename)).convert('RGB')
    img = np.asarray(img1) / 255.0
    R_channel[idx] = np.sum(img[:, :, 0])
    G_channel[idx] = np.sum(img[:, :, 1])
    B_channel[idx] = np.sum(img[:, :, 2])
    num += img.shape[0] * img.shape[1]
    print(str(idx+1))
 
# num = len(pathDir) * 240 * 320
# num = 86 * 1176 * 864 + (334 - 86) * 2352 * 1728
R_mean = np.sum(R_channel) / num
G_mean = np.sum(G_channel) / num
B_mean = np.sum(B_channel) / num
print('mean over!')
 
R_channel = np.zeros(len(pathDir))
G_channel = np.zeros(len(pathDir))
B_channel = np.zeros(len(pathDir))
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    # img = imread(os.path.join(filepath, filename)) / 255.0
    img1 = Image.open(os.path.join(filepath, filename)).convert('RGB')
    img = np.asarray(img1) / 255.0
    R_channel[idx] = np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel[idx] = np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel[idx] = np.sum((img[:, :, 2] - B_mean) ** 2)
    print(str(idx + 1))
 
R_var = np.sqrt(np.sum(R_channel) / num)
G_var = np.sqrt(np.sum(G_channel) / num)
B_var = np.sqrt(np.sum(B_channel) / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
