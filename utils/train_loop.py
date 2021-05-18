"""
Basic train loop.
"""

import torch


class Trainer:
    def __init__(self, net, train_loader, val_loader, optimizer, loss_fn, metric_fn, train_writer, val_writer, device=torch.device('cuda')):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

        self.train_writer = train_writer
        self.val_writer = val_writer

        self.device = device

    def _process_im(self, im):
        """
        Helper to process the input image for logging.
        """

        im = im - im.min()
        im = im / im.max()
        return im

    def _process_gt(self, gt):
        """
        Helper to process the ground truth for logging.
        """
        
        if gt.ndim == 3:
            gt = gt / gt.max()
            return gt.unsqueeze(1)
        else:
            gt = torch.argmax(gt, dim=1, keepdim=True)
            return gt / gt.max()

    def _process_pred(self, pred):
        """
        Helper to process the prediction for logging.
        """

        if pred.shape[1] == 2:
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            return pred / pred.max()
        else:
            return torch.sigmoid(pred)

    def __call__(self, curr_epoch=0):
        """
        Performs one epoch of optimization.
        """

        train_loss, train_metric, last_sample, last_pred = self._iterate_through_loader(self.train_loader, backpropagate=True)
        self.train_writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=curr_epoch)
        self.train_writer.add_scalar(tag='train_iou_background', scalar_value=train_metric[0], global_step=curr_epoch)
        self.train_writer.add_scalar(tag='train_iou_foreground', scalar_value=train_metric[1], global_step=curr_epoch)
        self.train_writer.add_images(tag='train_im1', img_tensor=self._process_im(last_sample['im1']), global_step=curr_epoch, dataformats='NCHW')
        self.train_writer.add_images(tag='train_im2', img_tensor=self._process_im(last_sample['im2']), global_step=curr_epoch, dataformats='NCHW')
        self.train_writer.add_images(tag='train_gt', img_tensor=self._process_gt(last_sample['gt']), global_step=curr_epoch, dataformats='NCHW')

        pred = self._process_pred(last_pred['pred'])
        if pred.shape[1] == 1:  # motion
            self.train_writer.add_images(tag='train_pred', img_tensor=pred, global_step=curr_epoch, dataformats='NCHW')
        else:
            self.train_writer.add_images(tag='train_background', img_tensor=pred[:, 0, :, :].unsqueeze(1), global_step=curr_epoch, dataformats='NCHW')
            self.train_writer.add_images(tag='train_foreground', img_tensor=pred[:, 1, :, :].unsqueeze(1), global_step=curr_epoch, dataformats='NCHW')
            self.train_writer.add_images(tag='train_manipulator', img_tensor=pred[:, 2, :, :].unsqueeze(1), global_step=curr_epoch, dataformats='NCHW')
            self.train_writer.add_images(tag='train_object', img_tensor=pred[:, 3, :, :].unsqueeze(1), global_step=curr_epoch, dataformats='NCHW')
            if pred.shape[1] == 5:
                self.train_writer.add_images(tag='train_distractor', img_tensor=pred[:, 4, :, :].unsqueeze(1), global_step=curr_epoch, dataformats='NCHW')
        if len(train_metric) != 2:
            self.train_writer.add_scalar(tag='train_iou_manipulator', scalar_value=train_metric[2], global_step=curr_epoch)
            self.train_writer.add_scalar(tag='train_iou_object', scalar_value=train_metric[3], global_step=curr_epoch)
        if len(train_metric) == 5:
            self.train_writer.add_scalar(tag='train_iou_distractor', scalar_value=train_metric[4], global_step=curr_epoch)

        val_loss, val_metric, last_sample, last_pred = self._iterate_through_loader(self.val_loader, backpropagate=False)
        self.val_writer.add_scalar(tag='val_loss', scalar_value=val_loss, global_step=curr_epoch)
        self.val_writer.add_scalar(tag='val_iou_background', scalar_value=val_metric[0], global_step=curr_epoch)
        self.val_writer.add_scalar(tag='val_iou_foreground', scalar_value=val_metric[1], global_step=curr_epoch)
        self.val_writer.add_images(tag='val_im1', img_tensor=self._process_im(last_sample['im1']),
                                     global_step=curr_epoch, dataformats='NCHW')
        self.val_writer.add_images(tag='val_im2', img_tensor=self._process_im(last_sample['im2']),
                                     global_step=curr_epoch, dataformats='NCHW')
        self.val_writer.add_images(tag='val_gt', img_tensor=self._process_gt(last_sample['gt']), global_step=curr_epoch,
                                     dataformats='NCHW')
        pred = self._process_pred(last_pred['pred'])
        if pred.shape[1] == 1:  # motion
            self.val_writer.add_images(tag='val_pred', img_tensor=pred, global_step=curr_epoch, dataformats='NCHW')
        else:
            self.val_writer.add_images(tag='val_background', img_tensor=pred[:, 0, :, :].unsqueeze(1),
                                         global_step=curr_epoch, dataformats='NCHW')
            self.val_writer.add_images(tag='val_foreground', img_tensor=pred[:, 1, :, :].unsqueeze(1),
                                         global_step=curr_epoch, dataformats='NCHW')
            self.val_writer.add_images(tag='val_manipulator', img_tensor=pred[:, 2, :, :].unsqueeze(1),
                                         global_step=curr_epoch, dataformats='NCHW')
            self.val_writer.add_images(tag='val_object', img_tensor=pred[:, 3, :, :].unsqueeze(1), global_step=curr_epoch,
                                         dataformats='NCHW')
            if pred.shape[1] == 5:
                self.val_writer.add_images(tag='val_distractor', img_tensor=pred[:, 4, :, :].unsqueeze(1), global_step=curr_epoch, dataformats='NCHW')
        if len(val_metric) != 2:
            self.val_writer.add_scalar(tag='val_iou_manipulator', scalar_value=val_metric[2], global_step=curr_epoch)
            self.val_writer.add_scalar(tag='val_iou_object', scalar_value=val_metric[3], global_step=curr_epoch)
        if len(val_metric) == 5:
            self.val_writer.add_scalar(tag='val_iou_distractor', scalar_value=val_metric[4], global_step=curr_epoch)
        
        # some printouts
        print(f"Finished epoch {curr_epoch}:")
        print(f"  Train loss: {train_loss:.2f}")
        print(f"  Train IoU background: {train_metric[0]:.2f}")
        print(f"  Train IoU foreground: {train_metric[1]:.2f}")
        if len(train_metric) != 2:
            print(f"  Train IoU manipulator: {train_metric[2]:.2f}")
            print(f"  Train IoU object: {train_metric[3]:.2f}")
        if len(train_metric) == 5:
            print(f"  Train IoU distractor: {train_metric[4]:.2f}")
        print(f"  Val loss: {val_loss:.2f}")
        print(f"  Val IoU background: {val_metric[0]:.2f}")
        print(f"  Val IoU foreground: {val_metric[1]:.2f}")
        if len(val_metric) != 2:
            print(f"  Val IoU manipulator: {val_metric[2]:.2f}")
            print(f"  Val IoU object: {val_metric[3]:.2f}")
        if len(val_metric) == 5:
            print(f"  Val IoU distractor: {val_metric[4]:.2f}")

    def _iterate_through_loader(self, loader, backpropagate=True):
        """
        Helper to iterate through one dataloader.
        """

        total_loss = 0.
        self.metric_fn.reset()

        if backpropagate:
            self.net = self.net.train()
        else:
            self.net = self.net.eval()

        for i, sample in enumerate(loader):
            if backpropagate:
                self.optimizer.zero_grad()

            for k, v in sample.items():
                sample[k] = sample[k].to(self.device)

            out = self.net(sample)
            loss = self.loss_fn(out['pred'], sample['gt'])

            if backpropagate:
                loss.backward()
                self.optimizer.step()

            self.metric_fn(out['pred'].detach(), sample['gt'])

            total_loss += loss.item()

        miou = self.metric_fn.get_iou(mean=False, zero_is_void=False)
        miou = [m.item() * 100 for m in miou]

        return total_loss / len(loader), miou, sample, out

