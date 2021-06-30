import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from mypath import Path
from PIL import Image
from dataloaders import make_data_loader
from modeling.deeplab import *
from dataloaders.utils import get_ce3tr_labels
from utils.metrics import Evaluator
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set the GPUs


class Tester(object):
    def __init__(self, args):
        if not os.path.isfile(args.model):
            raise RuntimeError("no checkpoint found at '{}'".format(args.model))
        self.args = args
        self.color_map = get_ce3tr_labels()
        self.test_loader, self.ids, self.nclass = make_data_loader(args)

        # Define model
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False,
                        pretrained=False)

        self.model = model
        device = torch.device('cpu')

        if self.args.no_cuda:
            checkpoint = torch.load(args.model, map_location=device)
        else:
            checkpoint = torch.load(args.model)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.evaluator = Evaluator(self.nclass)

    def save_image(self, array, id, op):
        '''
        save the RGB image
        '''
        text = 'gt'
        if op == 0:
            text = 'pred'
        file_name = str(id) + '_' + text + '.png'
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.nclass):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]

        rgb = np.dstack((r, g, b))

        save_img = Image.fromarray(rgb.astype('uint8'))
        save_img.save(self.args.save_path + os.sep + file_name)

    def blend_two_images(self, array, id):
        '''
        blend the segmentation result with original image
        :param array: segmentation result,array
        :return: save
        '''
        # transform original RGB image into RGBA
        basedir = Path.db_root_dir(self.args.dataset)
        img1 = Image.open(basedir+'image/'+str(id)+'.png')  # load the original image
        img1 = img1.convert('RGBA')

        # transform segmentation result into RGBA
        file_name = str(id) + '_pred.png'
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.nclass):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]

        rgb = np.dstack((r, g, b))

        img2 = Image.fromarray(rgb.astype('uint8'))
        # img2 = img2.resize((img1.width, img1.height), Image.NEAREST)
        img2 = img2.convert('RGBA')
        save_img = Image.blend(img1, img2, 0.4)

        save_img.save(self.args.save_path + os.sep + file_name)

    def test(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        if os.path.exists(self.args.save_path):
            shutil.rmtree(self.args.save_path)
        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)
        if self.args.no_resize:  # test original size of input images
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                with torch.no_grad():
                    output = self.model(image)
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                if self.args.no_blend:
                    self.save_image(pred[0], self.ids[i], 0)
                    self.save_image(target[0], self.ids[i], 1)
                else:
                    self.blend_two_images(pred[0], self.ids[i])

                self.evaluator.add_batch(target, pred)

            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            print('Testing:')
            print("Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwIoU: {:.4f}".format(Acc, Acc_class, mIoU, FWIoU))
        else:
            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                with torch.no_grad():
                    output = self.model(image)
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                # recover the size
                size = (target.shape[2], target.shape[1])
                pred = Image.fromarray(pred[0].astype('uint8'))
                pred = pred.resize(size, Image.NEAREST)
                pred = np.array(pred)
                target = target.squeeze(0)
                if self.args.no_blend:
                    self.save_image(pred, self.ids[i], 0)
                    self.save_image(target, self.ids[i], 1)
                else:
                    self.blend_two_images(pred, self.ids[i])
                self.evaluator.add_batch(target, pred)

            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            print('Testing:')
            print("Acc:{:.4f}, Acc_class:{:.4f}, mIoU:{:.4f}, fwIoU: {:.4f}".format(Acc, Acc_class, mIoU, FWIoU))


def main():
    parser = argparse.ArgumentParser(description='Pytorch RGTNet Test your data')
    parser.add_argument('--test', action='store_true', default=True,
                        help='test your data')
    parser.add_argument('--dataset', type=str, default='ce3tr',
                        choices=['pascal', 'coco', 'cityscapes', 'ce3tr'], help='dataset name')
    parser.add_argument('--backbone', default='resnet101',
                        choices=['resnet101', 'resnet50', 'xception', 'drn', 'mobilenet'],
                        help='what is your network backbone')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='output stride')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='image size')
    parser.add_argument('--model', type=str,
                        default='./run1/ce3tr/deeplab-resnet/experiment_0/model_best.pth',
                        help='load your model')
    parser.add_argument('--save_path', type=str,
                        default='./results/',
                        help='save your prediction data')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-resize', action='store_true', default=False,
                        help='test original size of input images or not')
    parser.add_argument('--no-blend', action='store_true', default=False,
                        help='blend the input and pred')

    args = parser.parse_args()
    print(args)
    if args.test:
        tester = Tester(args)
        tester.test()


if __name__ == "__main__":
    since = time.time()
    main()
    time_elapsed = time.time() - since
    print('Totally cost: {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
