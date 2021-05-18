"""
Confusion Matrix.
"""

import torch
from torch.nn.functional import softmax


class ConfusionMatrix:
    def __init__(self, num_classes, labels=None):
        self.num_classes = num_classes
        self.conf_mat = torch.zeros(num_classes, 3).cuda()
        self.labels = labels if labels is not None else list(range(num_classes))

    def reset(self):
        self.conf_mat = torch.zeros(self.num_classes, 3).cuda()

    def __call__(self, preds, gts):
        # expects:
        # gt:   B, H, W -- np.argmax(one_hot_encoded_label, 3)
        # pred: B, H, W -- np.argmax(softmax, 3)

        if preds.ndim == 4:
            preds = softmax(preds.detach(), dim=1)
            preds = torch.argmax(preds, dim=1, keepdim=True).squeeze()
        if gts.ndim == 4:
            gts = gts.squeeze(1)

        if gts.ndim == 2:
            gts = gts.unsqueeze(0)
        if preds.ndim == 2:
            preds = preds.unsqueeze(0)

        assert preds.ndim == 3
        assert gts.ndim == 3

        assert gts.max() < self.conf_mat.shape[0]
        assert preds.max() < self.conf_mat.shape[0]

        for b in range(gts.shape[0]):
            pred = preds[b]
            gt = gts[b]
            for i in range(self.conf_mat.shape[0]):
                temp = pred == i
                temp_l = gt == i

                # tp = np.logical_and(temp, temp_l)
                tp = temp & temp_l
                temp[temp_l] = True

                # fp = np.logical_xor(temp, temp_l)
                fp = temp ^ temp_l

                temp = pred == i
                temp[fp] = False

                # fn = np.logical_xor(temp, temp_l)
                fn = temp ^ temp_l

                self.conf_mat[i, 0] += torch.sum(tp)
                self.conf_mat[i, 1] += torch.sum(fp)
                self.conf_mat[i, 2] += torch.sum(fn)

    def get_iou(self, mean=False, zero_is_void=True):
        # intersection / union
        conf_mat = self.conf_mat.clone()
        union = torch.sum(conf_mat, dim=1)
        if zero_is_void:
            iou = conf_mat[1:, 0] / (union[1:] + 1e-10)
        else:
            iou = conf_mat[:, 0] / (union + 1e-10)

        if mean:
            iou = torch.sum(iou) / iou.shape[0]
        return iou

    def get_precision(self, mean=False, zero_is_void=True):
        # tp / (tp + fp)
        conf_mat = self.conf_mat.clone()

        tp_fp = conf_mat[:, 0] + conf_mat[:, 1]
        if zero_is_void:
            pre = conf_mat[1:, 0] / (tp_fp[1:] + 1e-10)
        else:
            pre = conf_mat[:, 0] / (tp_fp + 1e-10)

        if mean:
            pre = torch.sum(pre) / pre.shape[0]

        return pre

    def get_recall(self, mean=False, zero_is_void=True):
        # tp / (tp + fn)
        conf_mat = self.conf_mat.clone()

        tp_fn = conf_mat[:, 0] + conf_mat[:, 2]
        if zero_is_void:
            rec = conf_mat[1:, 0] / (tp_fn[1:] + 1e-10)
        else:
            rec = conf_mat[:, 0] / (tp_fn + 1e-10)

        if mean:
            rec = torch.sum(rec) / rec.shape[0]

        return rec

    def get_f1_score(self, mean=False, zero_is_void=True):
        pre = self.get_precision(mean=mean, zero_is_void=zero_is_void)
        rec = self.get_recall(mean=mean, zero_is_void=zero_is_void)

        f1 = 2 * ((pre * rec) / (pre + rec))

        return f1


class ConfusionMatrixSemantic(ConfusionMatrix):
    def __init__(self, num_classes, labels=None):
        super().__init__(num_classes=num_classes, labels=labels)

    def __call__(self, preds, gts):
        preds = torch.sigmoid(preds)
        for b in range(gts.shape[0]):
            pred = preds[b]
            gt = gts[b]
            for i in range(self.conf_mat.shape[0]):
                temp = pred[i] >= 0.5
                temp_l = gt[i] == 1

                # tp = np.logical_and(temp, temp_l)
                tp = temp & temp_l
                temp[temp_l] = True

                # fp = np.logical_xor(temp, temp_l)
                fp = temp ^ temp_l

                temp = pred[i] >= 0.5
                temp[fp] = False

                # fn = np.logical_xor(temp, temp_l)
                fn = temp ^ temp_l

                self.conf_mat[i, 0] += torch.sum(tp)
                self.conf_mat[i, 1] += torch.sum(fp)
                self.conf_mat[i, 2] += torch.sum(fn)

