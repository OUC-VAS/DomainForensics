import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.transformer.ssrt import SSRT, SSRTRestoreManager
from models.transformer.ssrt_con import SSRT
from models.components.resnet import ResnetBase, ResClassifierMME
from models.components.discriminator import Discriminator
from models.components.xception import return_pytorch04_xception
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from configs.defaults import CfgNode
from models.components.effcientnet import get_efficientnet


class BaseFaceAdaptation(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseFaceAdaptation, self).__init__()

        self.save_hyperparameters()
        feat_dim = self.hparams.cfg.MODEL.CLASSIFICATION_FEATURE
        self.model = SSRT(base_net='vit_base_patch16_224', bottleneck_dim=feat_dim, class_num=2, cfg=cfg)
        self.backbone = self.model.net

        self.discriminator = Discriminator(input_dim=feat_dim * 2, hidden_dim=self.hparams.cfg.MODEL.DISCRIMINATOR_MIDDIM, out_dim=1)
        self.gobal_i = 0
        self.automatic_optimization = False
        self.use_proto_align = False
        self.local_alignment = self.hparams.cfg.MODEL.LOCAL_ALIGNMENT
        self.cycle_flag = True

    def forward(self, x):
        _, out, _ = self.backbone(x)
        return out

    def merge_from_other_cfg(self, cfg:CfgNode):
        self.hparams.cfg.merge_from_other_cfg(cfg)
        self.hparams.cfg.freeze()

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            print('Transfer from ', self.hparams.cfg.DATAS.SOURCE, ' to ----------- ', self.hparams.cfg.DATAS.TARGET)
        # define feature queue only for target domain
        if self.local_alignment:
            self.queue_size = self.hparams.cfg.MODEL.QUEUE_SIZE
            proj_dim = self.hparams.cfg.MODEL.CLASSIFICATION_FEATURE

            self.queue_target = torch.zeros(
                self.queue_size, proj_dim
            ).cuda()
            self.queue_ptr = torch.zeros(1, dtype=torch.long).cuda()
            self.queue_target.require_grad = False
            self.queue_ptr.require_grad = False
        #
        if self.hparams.cfg.DOMAIN_FINETUNING.ENABLE:
            feat_dim = self.hparams.cfg.MODEL.CLASSIFICATION_FEATURE
            print("Creating the Snap model for self-distillation ... ")
            self.model_snap = SSRT(base_net='vit_base_patch16_224', bottleneck_dim=feat_dim, class_num=2, cfg=self.hparams.cfg)
            self.model_snap.net.load_state_dict(self.backbone.state_dict())
            self.model_snap.net.cuda().eval()

    def _concat_and_gather(self, tensor):
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def _enqueue_and_dequeue(self, feat_target_aug): #, feat_source_aug
        feat_target_aug = self._concat_and_gather(feat_target_aug)
        feat_target_aug = F.normalize(feat_target_aug)
        tfn = feat_target_aug.size()[0]
        t_ptr = int(self.queue_ptr)
        self.queue_target[t_ptr:t_ptr + tfn, :] = feat_target_aug
        t_ptr = (t_ptr + tfn) % self.queue_size
        self.queue_ptr[0] = t_ptr

    def domain_finetume(self, s_x, s_y, t_x, t_y, s_ycbcr=None, t_ycbcr=None):
        # data
        # tx - t_y~
        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            opt, opt_d = self.optimizers()
        else:
            opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if t_ycbcr is not None:
                # x = torch.cat((s_x, t_x), dim=0)
                # ycbcr = torch.cat((s_ycbcr, t_ycbcr), dim=0)
                # features, outputs = self.model_snap.net(x, ycbcr)
                features, outputs = self.model_snap.net(t_x, t_ycbcr)
            else:
                features, outputs = self.model_snap.net(t_x)

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
                if self.use_proto_align:
                    total_loss, distill_loss, disc_loss, proto_loss = self.model.get_finetune_loss(s_x=s_x, s_y=s_y, t_x=t_x,
                                                                                       t_y=outputs,
                                                                                       s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr,
                                                                                       discriminator=self.discriminator)
                else:
                    total_loss, distill_loss, disc_loss = self.model.get_finetune_loss(s_x=s_x, s_y=s_y, t_x=t_x, t_y=outputs,
                                                                                   s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr,
                                                                                   discriminator=self.discriminator)
            else:
                total_loss, distill_loss = self.model.get_finetune_loss(s_x=s_x, s_y=s_y, t_x=t_x, t_y=outputs,
                                                                        s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr)
        else:
            if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
                total_loss, distill_loss, disc_loss = self.model.get_finetune_loss(s_x=s_x, s_y=s_y, t_x=t_x, t_y=outputs,
                                                                               discriminator=self.discriminator)
            else:
                total_loss, distill_loss = self.model.get_finetune_loss(s_x=s_x, s_y=s_y, t_x=t_x, t_y=outputs)
        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()
        # single scheduler
        sch = self.lr_schedulers()
        sch.step()

        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            opt_d.zero_grad()
            self.discriminator.zero_grad()
            if self.hparams.cfg.DATAS.WITH_FREQUENCY:
                discriminator_loss = self.model.discriminator_loss(s_x, t_x, self.discriminator, s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr)
            else:
                discriminator_loss = self.model.discriminator_loss(s_x, t_x, self.discriminator)
            self.manual_backward(discriminator_loss)
            opt_d.step()
            opt_d.zero_grad()
            # sch_d.step()

        self.log('distill_loss', float(distill_loss.detach().cpu()), on_step=True)
        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            self.log('disc_loss', float(disc_loss.detach().cpu()), on_step=True)
        return total_loss

    def on_epoch_end(self) -> None:
        if self.current_epoch > 1:
            # update the snap model
            print("Update model snap")
            self.model_snap.net.load_state_dict(self.backbone.state_dict())
            self.model_snap.net.eval()

    def training_step(self, batch, batch_idx):

        self.gobal_i += 1
        batch_s, batch_t = batch['source'], batch['target']

        # source
        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            s_x, s_ycbcr, s_y = batch_s
            t_x, t_ycbcr, t_y = batch_t
        else:
            s_x, s_y = batch_s
            t_x, t_y = batch_t

        if self.hparams.cfg.DOMAIN_FINETUNING.ENABLE:
            if self.hparams.cfg.DATAS.WITH_FREQUENCY:
                return self.domain_finetume(s_x=s_x, s_y=s_y, t_x=t_x, t_y=t_y, s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr)
            else:
                return self.domain_finetume(s_x=s_x, s_y=s_y, t_x=t_x, t_y=t_y)

        opt, opt_d = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if t_ycbcr is not None:
                features, outputs = self.model_snap.net(t_x, t_ycbcr)
            else:
                features, outputs = self.model_snap.net(t_x)

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            total_loss, cls_loss, disc_loss, distill_loss, disc_tos_loss = \
                self.model.get_loss(inputs_source=s_x, inputs_target=t_x, labels_target=outputs,
                                    labels_source=s_y, inputs_s_ycbcr=s_ycbcr,
                                    inputs_t_ycbcr=t_ycbcr, discriminator=self.discriminator,
                                    cur_epoch=self.current_epoch)
        else:
            total_loss, cls_loss, disc_loss = self.model.get_loss(inputs_source=s_x, inputs_target=t_x,
                                                                  labels_source=s_y, discriminator=self.discriminator,
                                                                  cur_epoch=self.current_epoch)


            # domain finetune loss
            # total_ft_loss, distill_ft_loss, disc_ft_loss = self.model.get_finetune_loss(s_x=s_x, s_y=s_y, t_x=t_x, t_y=outputs,
            #                                                                             s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr,
            #                                                                             discriminator=self.discriminator)
        # if self.cycle_flag:
        #     total_loss += 0.1 * total_ft_loss
        # else:
        #     total_loss = 0.1 * total_loss + total_ft_loss

        total_loss += 0.1 * (distill_loss + disc_tos_loss)
        self.manual_backward(total_loss)
        opt.step()
        # single scheduler
        sch = self.lr_schedulers()
        sch.step()

        opt_d.zero_grad()
        self.discriminator.zero_grad()
        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            discriminator_loss = self.model.discriminator_loss(s_x, t_x, self.discriminator, s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr, cur_epoch=self.current_epoch)
        else:
            discriminator_loss = self.model.discriminator_loss(s_x, t_x, self.discriminator, cur_epoch=self.current_epoch)
        self.manual_backward(discriminator_loss)
        opt_d.step()
        opt_d.zero_grad()
        opt.zero_grad()

        del discriminator_loss
        del s_x
        del t_x

        self.log('ce_loss', float(cls_loss.detach().cpu()), on_step=True)
        self.log('disc_loss', float(disc_loss.detach().cpu()), on_step=True)
        self.log('distill_loss', float(distill_loss.detach().cpu()), on_step=True)

        return total_loss

    def configure_optimizers(self):
        params = self.model.get_parameter_list()

        if not self.hparams.cfg.DOMAIN_FINETUNING.ENABLE:
            if self.hparams.cfg.TRAINING.SCHEDULER == 'cosine':
                for group in params:
                    group['lr'] = group['lr'] * self.hparams.cfg.TRAINING.LR

            optim = torch.optim.SGD(
                params, lr=self.hparams.cfg.TRAINING.LR, momentum=0.9, nesterov=True, weight_decay=0.0005
            )
            opt_d = torch.optim.SGD(
                self.discriminator.parameters(), lr=self.hparams.cfg.TRAINING.LR, momentum=0.9, nesterov=True, weight_decay=0.0005
            )

            if self.hparams.cfg.TRAINING.SCHEDULER == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=5, eta_min=1e-4)
                # scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_d, T_max=5000, eta_min=1e-5)
            else:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: self.hparams.cfg.TRAINING.LR * (1. + 0.001 * float(x)) ** (-0.75))
                # scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda x: self.hparams.cfg.TRAINING.LR * (
                #         1. + 0.001 * float(x)) ** (-0.75))
        else:

            if self.hparams.cfg.TRAINING.SCHEDULER == 'cosine':
                for group in params:
                    group['lr'] = group['lr'] * self.hparams.cfg.DOMAIN_FINETUNING.LR
            optim = torch.optim.SGD(
                params, lr=self.hparams.cfg.DOMAIN_FINETUNING.LR*0.1, momentum=0.9, nesterov=True, weight_decay=0.0005
            )
            if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
                opt_d = torch.optim.SGD(
                    self.discriminator.parameters(), lr=self.hparams.cfg.DOMAIN_FINETUNING.LR * 0.1,
                    momentum=0.9, nesterov=True, weight_decay=0.0005
                )
            if self.hparams.cfg.TRAINING.SCHEDULER == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=5000, eta_min=1e-5)
                # scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt_d, T_max=5000, eta_min=1e-5)
            else:
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: self.hparams.cfg.TRAINING.LR * (1. + 0.001 * float(x)) ** (-0.75))
                # scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda x: self.hparams.cfg.TRAINING.LR * (
                #         1. + 0.001 * float(x)) ** (-0.75))
        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            return [optim, opt_d], [scheduler] #, opt_d
        else:
            return [optim], [scheduler]

    def _model_predict(self, batch, batch_idx, mode='train'):

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            x, freq_x, y = batch
        else:
            x, y = batch
            freq_x = None
        if mode == 'test':
            pred_labels = []
            pred_features = []
            pred_scores = []
            for i in range(x.size()[0]):
                if self.hparams.cfg.DATAS.WITH_FREQUENCY:
                    feat, y_hat = self.backbone(x[i], freq_x[i])
                else:
                    feat, y_hat = self.backbone(x[i])
                y_hat = F.softmax(y_hat, dim=-1)
                scores = y_hat[:, 1]
                score = torch.mean(scores)
                pred_scores.append(score)
                if score > 0.5:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)
                # pred = preds[torch.argmax(scores)]
                # pred_labels.append(pred)
                pred_features.append(feat.unsqueeze(0))
            pred_labels = torch.tensor(pred_labels)
            pred_scores = torch.tensor(pred_scores)
            pred_features = torch.cat(pred_features, dim=0)
            return {
                'scores': pred_scores,
                'pred': pred_labels,
                'label': y,
                'feat': pred_features,
            }
        else:
            feat, y_hat = self.backbone(x, freq_x)
            y_hat = F.softmax(y_hat, dim=-1)
            # pred = torch.max(y_hat, dim=1)[1]
            scores, preds = torch.max(y_hat, dim=1)
            scores = y_hat[:, 1]
            pred = (scores > 0.5).int()
            return {
                'scores': scores,
                'pred': pred,
                'label': y,
                'feat': feat,
            }

    def _eval_results(self, outputs, save_pth=False, mode='train'):
        # TODO: complete auc and other evaluation metric
        all_preds = torch.cat([_['pred'] for _ in outputs])
        all_labels = torch.cat([_['label'] for _ in outputs])
        all_scores = torch.cat([_['scores'] for _ in outputs])
        if save_pth:
            all_feat = torch.cat([_['feat'] for _ in outputs])
            data = {
                'label': all_labels,
                'scores': all_scores,
                'pred': all_preds,
                'feat': all_feat
            }
            torch.save(data, './outs/pred.pth')
        try:
            acc = accuracy_score(all_labels.detach().cpu().numpy(), all_preds.detach().cpu().numpy())
            auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_scores.detach().cpu().numpy())
            self.log("accuracy", float(acc)*100, on_epoch=True)
            self.log("auc", float(auc)*100, on_epoch=True)
        except:
            self.log("accuracy", 0.0, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.backbone.eval()
        return self._model_predict(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        self._eval_results(validation_step_outputs, save_pth=False)
        self.backbone.train()

    def test_step(self, batch, batch_idx):
        # Testing on video level

        self.backbone.eval()
        return self._model_predict(batch, batch_idx, mode='test')

    def test_epoch_end(self, test_step_outputs):
        self._eval_results(test_step_outputs, save_pth=True)


