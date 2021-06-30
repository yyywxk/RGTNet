import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
 
        intersection = input_flat * target_flat
 
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
 
        return loss


class MulticlassDiceLoss(nn.Module):
    '''
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    '''
    def __init__(self, n_classes, cuda):
        super(MulticlassDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.cuda = cuda

    @staticmethod
    def to_one_hot(tensor, n_classes, cuda):
        n, h, w = tensor.size()
        if cuda:
            one_hot = torch.zeros(n, n_classes, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
        else:
            one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot
 
    def forward(self, input, target, weights=None):
        # logit => N x Classes x H x W
        # target => N x H x W
        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes, self.cuda)
        C = target_onehot.shape[1]

 
        # C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
 
        dice = DiceLoss()
        totalLoss = 0
 
        for i in range(C):
            # diceLoss = dice(input[:, i], target[:, i])
            diceLoss = dice(pred[:, i], target_onehot[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
 
        return totalLoss


# Multiclass Smooth IOU loss
class SoftIoULoss(nn.Module):
    def __init__(self, n_classes, cuda):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.cuda = cuda

    @staticmethod
    def to_one_hot(tensor, n_classes, cuda, label_smoothing=1e-5):
        n, h, w = tensor.size()
        if cuda:
            one_hot = torch.zeros(n, n_classes, h, w).cuda().scatter_(1, tensor.view(n, 1, h, w), 1)
        else:
            one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)

        one_hot = one_hot * (1 - label_smoothing) + label_smoothing / n_classes  # label smoothing
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes, self.cuda)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        # self.weight = torch.Tensor(np.array([0.1, 1.0]))
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='dice'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'dice':
            return self.Dice_Loss
        elif mode == 'iou':
            return self.IouLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)

        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def Dice_Loss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = MulticlassDiceLoss(n_classes=c, cuda=self.cuda)

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def IouLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = SoftIoULoss(n_classes=c, cuda=self.cuda)

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    # loss = SegmentationLosses(cuda=True)
    # a = torch.rand(1, 3, 7, 7).cuda()
    # b = torch.rand(1, 7, 7).cuda()
    # print(loss.CrossEntropyLoss(a, b).item())
    # print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    # print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    # print(loss.Dice_Loss(a, b).item())
    # print(loss.IouLoss(a, b).item())

    loss = SegmentationLosses(cuda=False)
    a = torch.rand(1, 3, 7, 7)
    b = torch.rand(1, 7, 7)
    b[b > 0.7] = 2
    b[b <= 0.3] = 1
    b[b < 1] = 0
    # print(loss.Dice_Loss(a, b.long()).item())
    print(loss.IouLoss(a, b.long()).item())
