# To obtain image names into a text
import random
import glob

img_path = glob.glob('./image/*.png')
TrainPath = './index/train.txt'
ValPath = './index/val.txt'
TestPath = './index/test.txt'

# get whole set
for each in img_path:
    with open('./index/all.txt', 'a')as f:
        f.write(each[8:-4]+'\n')

# get training and validation set
allnum = 334
trnum = int(allnum*0.6)
vanum = int(allnum*0.8)
with open('./index/all.txt', 'r')as f:
    lines = f.readlines()
    g = [i for i in range(allnum)]
    random.shuffle(g)
    # train:val:test = 6:2:2
    train = g[:trnum]
    trainval = g[trnum:vanum]
    val = g[vanum:]

    for index, line in enumerate(lines,1):
        if index-1 in train:
            with open(TrainPath,'a')as trainf:
                trainf.write(line)
        elif index-1 in trainval:
            with open(ValPath,'a')as trainvalf:
                trainvalf.write(line)
        elif index-1 in val:
            with open(TestPath,'a')as valf:
                valf.write(line)
