class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return './VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return './datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return './datasets/leftImg8bit_trainvaltest/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return './datasets/coco/'
        elif dataset == 'ce3tr':
            return './datasets/ce3tr/train/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError