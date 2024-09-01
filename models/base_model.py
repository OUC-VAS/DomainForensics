import torch
import torch.nn.functional as F

from typing import Any, Optional, Dict
from pytorch_lightning.core.lightning import LightningModule
from models.components import ResnetBase, ResClassifierMME

# lr scheduler
def lambda_func(step):
    gamma = 10
    power = 0.75
    max_iter = 30000
    return (1 + gamma * min(1.0, step / max_iter)) ** (-power)


class ResNetCL(LightningModule):
    def __init__(self, model_params, hyper_params, dst_params):
        super(ResNetCL, self).__init__()
        self.save_hyperparameters()
        # model backbone
        self.backbone = ResnetBase(model=self.hparams.model_params['base_model'], pretrained=True)
        # classifier
        self.classifier = ResClassifierMME(num_classes=self.hparams.hyper_params['num_classes'], input_size=2048)

        self.threshold = 7.0

        self.automatic_optimization = False

    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)
        return out

    def on_train_start(self) -> None:
        if torch.cuda.device_count() > 1:
            print("Let's use ", torch.cuda.device_count(), " GPUs!")
            self.source_size = self.trainer.train_dataloader.sampler['source'].loader.sampler.total_size
            self.batch_size = self.trainer.train_dataloader.sampler['source'].loader.batch_size
            self.source_batch_nums = self.source_size // self.batch_size + 1
            self.source_epoch = 0

            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
            gpu = self.local_rank
            self.backbone = torch.nn.parallel.DistributedDataParallel(self.backbone, device_ids=[gpu])
            self.classifier = torch.nn.parallel.DistributedDataParallel(self.classifier, device_ids=[gpu])
        else:
            print("Using only one GPU")

        return super().on_train_start()

    def training_step(self, batch, batch_idx):

        opts = self.optimizers()
        for opt in opts:
            opt.zero_grad()

        # Train
        data_s, data_t = batch['source'], batch['target']
        img_s, label_s, _ = data_s

        # img_t, img_t_masked, label_t = data_t

        img_s, label_s = img_s.cuda(), label_s.cuda()

        self.classifier.module.weight_norm()

        feat = self.backbone(img_s)

        out_s = self.classifier(feat)
        loss_s = F.cross_entropy(out_s, label_s)
        self.log('ce_loss', loss_s, on_step=True)
        # add global step into progress bar
        self.log("Step", float(self.global_step), prog_bar=True)

        loss = loss_s
        self.manual_backward(loss)

        for opt in opts:
            opt.step()

    def validation_step(self, batch, batch_idx):

        pred, label_t = self._infer_test(batch, threshold=7.0)

        return {
            'label_target': label_t,
            'label_pred': pred
        }

    def validation_epoch_end(self, outputs):
        log_data = self._process_infer_output(outputs)
        self.log('H-score', float(log_data["H score"]), on_epoch=True)
        self.log('Mean Acc', float(log_data["Com Mean Acc"]), on_epoch=True)
        self.log('Unknow Acc', float(log_data["Unk Acc"]), on_epoch=True)
        print(log_data)

    def _process_infer_output(self, outputs):
        target_lbls = torch.cat([_['label_target'] for _ in outputs])
        pred_lbls = torch.cat([_['label_pred'] for _ in outputs])

        n_total = target_lbls.size()[0]
        correct = (pred_lbls == target_lbls).sum()
        total_accuracy = correct / n_total

        class_list = torch.unique(target_lbls)
        class_accs = []
        for c in class_list:
            labeled_mask = target_lbls == c
            labeled_c_gt = target_lbls[labeled_mask]
            labeled_c_pred = pred_lbls[labeled_mask]

            c_acc = (labeled_c_pred == labeled_c_gt).sum() / labeled_c_gt.size()[0]
            class_accs.append(float(c_acc))

        unk_mask = target_lbls == self.hparams.hyper_params['num_classes']
        gt_unk = target_lbls[unk_mask]
        pred_unk = pred_lbls[unk_mask]
        fp_rate = (pred_unk != self.hparams.hyper_params['num_classes']).sum() / gt_unk.size()[0]

        class_accs = torch.tensor(class_accs)

        hscore = 2 * (class_accs[:-1].mean() * class_accs[-1]) / (class_accs[:-1].mean() + class_accs[-1])
        log_data = {
            "epoch": self.global_step,
            "Per Class Acc": class_accs,
            "Total Acc": total_accuracy,
            "Mean Acc": class_accs.mean(),
            "Com Mean Acc": class_accs[:-1].mean(),
            "Unk Acc": class_accs[-1],
            "H score": hscore,
            "FP Rate": fp_rate
        }
        return log_data

    def _infer_test(self, batch, threshold, eng_mean_std=False):
        with torch.no_grad():
            img_t, label_t, _ = batch
            img_t = img_t.cuda()
            label_t = label_t.cuda()
            features = self.backbone(img_t)
            out_t = self.classifier(features)
            energy = torch.logsumexp(out_t, dim=1).data.cpu()
            # p = F.softmax(out_t, dim=1)
            # entropy = - torch.sum(p * torch.log(p + 1e-5), dim=1).cpu()  # [nt, ]
            pred = out_t.max(1)[1]
            pred_unk = energy < threshold
            # pred_unk = entropy > 0.9
            pred[pred_unk] = self.hparams.hyper_params['num_classes']
        if eng_mean_std:
            return pred, label_t, energy, features
        return pred, label_t

    def test_step(self, batch, batch_idx):
        pred, label_t, energy, features = self._infer_test(batch, self.infer_threshold, eng_mean_std=True)

        return {
            'label_target': label_t,
            'label_pred': pred,
            'energy': energy,
            'features': features
        }

    def on_test_start(self) -> None:
        self.backbone.eval()
        self.classifier.eval()
        print(" Start testing  ...............")

    def configure_optimizers(self):
        optim_backbone = torch.optim.SGD(
            self.backbone.parameters(), lr=self.hparams.hyper_params['lr'] * 0.1,
            momentum=self.hparams.hyper_params['sgd_mom'],
            weight_decay=0.0005, nesterov=True
        )
        optim_classifier = torch.optim.SGD(
            self.classifier.parameters(), lr=self.hparams.hyper_params['lr'],
            momentum=self.hparams.hyper_params['sgd_mom'],
            weight_decay=self.hparams.hyper_params['weight_decay'], nesterov=True
        )

        backbone_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim_backbone, lambda_func)
        classifier_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim_classifier, lambda_func)

        return [optim_backbone, optim_classifier], [backbone_lr_scheduler, classifier_lr_scheduler]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        state_dict_old = checkpoint['state_dict']
        state_dict_new = {}
        for k, v in state_dict_old.items():
            if 'backbone' or 'classifier' in k:
                new_k = k.split('.')
                new_k = [new_k[0]] + new_k[2:]
                new_k = '.'.join(new_k)
                state_dict_new[new_k] = state_dict_old[k]
            else:
                state_dict_new[k] = state_dict_old[k]
        checkpoint['state_dict'] = state_dict_new
        return super().on_save_checkpoint(checkpoint)