from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr


class Ce3trSegmentation(Dataset):
    """
    ce3tr dataset
    """
    NUM_CLASSES = 3

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('ce3tr'),
                 split='train',
                 ):
        """
        :param base_dir: path to change dataset directory
        :param split: train/val/test
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'image')
        self._cat_dir = os.path.join(self._base_dir, 'mask')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir, 'index')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".png")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)
            elif split == 'test':
                return self.transform_te(sample)

    def _make_img_gt_point_pair(self, index):
        # original input
        # _img = Image.open(self.images[index]).convert('RGB')

        # LBP input
        name = self.images[index]
        name1 = self._image_dir + '1/' + name[len(self._image_dir)+1:]
        name2 = self._image_dir + '2/' + name[len(self._image_dir)+1:]
        img = Image.open(name).convert('L')
        img1 = Image.open(name1).convert('L')
        img2 = Image.open(name2).convert('L')
        _img = Image.merge('RGB', (img, img1, img2))

        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            # tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_te(self, sample):
        if self.args.no_resize:  # test original size of input images
            composed_transforms = transforms.Compose([
                # tr.FixScaleCrop(crop_size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        else:  # test resized input images
            composed_transforms = transforms.Compose([
                # tr.FixScaleCrop(crop_size=self.args.crop_size),
                tr.ResizeImage(size=self.args.crop_size),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Ce3tr(split=' + str(self.split) + ')'
