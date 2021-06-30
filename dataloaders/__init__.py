from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd, ce3tr
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'pascal' and not args.test:
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'pascal' and args.test:
        test_set = pascal.VOCSegmentation(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        ids = test_set.im_ids
        return test_loader, ids, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'ce3tr':
        if not args.test:  # train mode
            train_set = ce3tr.Ce3trSegmentation(args, split='train')
            val_set = ce3tr.Ce3trSegmentation(args, split='val')

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None

            return train_loader, val_loader, test_loader, num_class
        else:  # test mode
            test_set = ce3tr.Ce3trSegmentation(args, split='test')
            num_class = test_set.NUM_CLASSES
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
            ids = test_set.im_ids
            return test_loader, ids, num_class

    else:
        raise NotImplementedError
