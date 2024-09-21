import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.transformer.ssrt import SSRT, SSRTRestoreManager
# from models.transformer.ssrt_con import SSRT
from models.transformer.banet import SSRT
from models.components.discriminator import Discriminator
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from configs.defaults import CfgNode
from models.components.xception import return_pytorch04_xception, XceptionModel
from models.transfer_base import TransferModel


class BaseFaceAdaptation(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseFaceAdaptation, self).__init__()

        self.save_hyperparameters()
        self.model = TransferModel(cfg=cfg)

        feat_dim = self.model.classifier_dim
        self.backbone = self.model.net

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            disc_dim = feat_dim + 1024
        else:
            disc_dim = feat_dim
        self.discriminator = Discriminator(input_dim=disc_dim, hidden_dim=self.hparams.cfg.MODEL.DISCRIMINATOR_MIDDIM, out_dim=1)
        # self.discriminator = None
        self.gobal_i = 0
        self.automatic_optimization = False
        self.use_proto_align = False
        self.local_alignment = self.hparams.cfg.MODEL.LOCAL_ALIGNMENT

    def forward(self, x):
        _, out, _ = self.backbone(x)
        return out

    def merge_from_other_cfg(self, cfg:CfgNode):
        self.hparams.cfg.merge_from_other_cfg(cfg)
        self.hparams.cfg.freeze()

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            print('Transfer from ', self.hparams.cfg.DATAS.SOURCE, ' to ----------- ', self.hparams.cfg.DATAS.TARGET)
        if self.hparams.cfg.DOMAIN_FINETUNING.ENABLE:
            print("Creating the Snap model for self-distillation ... ")
            self.model_snap = TransferModel(cfg=self.hparams.cfg)
            self.model_snap.net.load_state_dict(self.backbone.state_dict())
            self.model_snap.net.cuda().eval()

    def domain_finetume(self, s_x, s_y, t_x, t_y, s_ycbcr=None, t_ycbcr=None):
        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            opt, opt_d = self.optimizers()
        else:
            opt = self.optimizers()
        opt.zero_grad()

        with torch.no_grad():
            if t_ycbcr is not None:
                features, outputs = self.model_snap.net(t_x, t_ycbcr)
            else:
                features, outputs = self.model_snap.net(t_x)

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:

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

        self.log('distill_loss', float(distill_loss.detach().cpu()), on_step=True)
        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            self.log('disc_loss', float(disc_loss.detach().cpu()), on_step=True)
        return total_loss

    def transfer_training(self, s_x, s_y, t_x, t_y, s_ycbcr=None, t_ycbcr=None):
        opt, opt_d = self.optimizers()
        # opt = self.optimizers()
        opt.zero_grad()

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            total_loss, cls_loss, disc_loss = self.model.get_loss(inputs_source=s_x, inputs_target=t_x,
                                                                  labels_source=s_y, inputs_s_ycbcr=s_ycbcr,
                                                                  inputs_t_ycbcr=t_ycbcr,
                                                                  discriminator=self.discriminator,
                                                                  cur_epoch=self.current_epoch)
        else:
            total_loss, cls_loss, disc_loss = self.model.get_loss(inputs_source=s_x, inputs_target=t_x,
                                                                  labels_source=s_y, discriminator=self.discriminator,
                                                                  cur_epoch=self.current_epoch)

        self.manual_backward(total_loss)
        # cls_loss = total_loss.detach()
        opt.step()
        # single scheduler
        # sch = self.lr_schedulers()
        # sch.step()

        opt_d.zero_grad()
        self.discriminator.zero_grad()
        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            discriminator_loss = self.model.discriminator_loss(s_x, t_x, self.discriminator, s_ycbcr=s_ycbcr,
                                                               t_ycbcr=t_ycbcr, cur_epoch=self.current_epoch)
        else:
            discriminator_loss = self.model.discriminator_loss(s_x, t_x, self.discriminator,
                                                               cur_epoch=self.current_epoch)
        self.manual_backward(discriminator_loss)
        opt_d.step()
        opt_d.zero_grad()
        opt.zero_grad()

        # del discriminator_loss
        del s_x
        del t_x

        self.log('ce_loss', float(cls_loss.detach().cpu()), on_step=True)
        self.log('disc_loss', float(disc_loss.detach().cpu()), on_step=True)
        return total_loss

    def on_train_epoch_end(self) -> None:
        if self.hparams.cfg.DOMAIN_FINETUNING.ENABLE:
            if self.current_epoch > 1:
                # update the snap model
                print("Update model snap")
                self.model_snap.net.load_state_dict(self.backbone.state_dict())
                self.model_snap.net.eval()

    def training_step(self, batch, batch_idx):
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
        else:
            if self.hparams.cfg.DATAS.WITH_FREQUENCY:
                return self.transfer_training(s_x=s_x, s_y=s_y, t_x=t_x, t_y=t_y, s_ycbcr=s_ycbcr, t_ycbcr=t_ycbcr)
            else:
                return self.transfer_training(s_x=s_x, s_y=s_y, t_x=t_x, t_y=t_y)

    def configure_optimizers(self):
        params = self.model.get_parameter_list()
        for p in params:
            p['lr'] = 0.0002
        if not self.hparams.cfg.DOMAIN_FINETUNING.ENABLE:
            optim = torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-8)

            opt_d = torch.optim.SGD(
                self.discriminator.parameters(), lr=self.hparams.cfg.TRAINING.LR, momentum=0.9, nesterov=True, weight_decay=0.0005
            )
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: self.hparams.cfg.TRAINING.LR * (1. + 0.001 * float(x)) ** (-0.75))
        else:
            optim = torch.optim.SGD(
                params, lr=self.hparams.cfg.DOMAIN_FINETUNING.LR * 0.1, momentum=0.9, nesterov=True, weight_decay=0.0005
            )
            if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
                opt_d = torch.optim.SGD(
                    self.discriminator.parameters(), lr=self.hparams.cfg.DOMAIN_FINETUNING.LR * 0.1,
                    momentum=0.9, nesterov=True, weight_decay=0.0005
                )
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: self.hparams.cfg.TRAINING.LR * (1. + 0.001 * float(x)) ** (-0.75))

        if self.hparams.cfg.DOMAIN_FINETUNING.DISCRIMINATOR:
            return [optim, opt_d]#, [scheduler] #, opt_d
        else:
            return [optim]#, [scheduler]

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

            if len(x.size()) == 4:
                if self.hparams.cfg.DATAS.WITH_FREQUENCY:
                    feat, y_hat = self.backbone(x, freq_x)
                else:
                    feat, y_hat = self.backbone(x)
                y_hat = F.softmax(y_hat, dim=-1)
                scores = y_hat[:, 1]
                pred = (scores > 0.5).int()
                return {
                    'scores': scores,
                    'pred': pred,
                    'label': y,
                    'feat': feat,
                }
            else:
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
            # all_feat = torch.cat([_['feat'] for _ in outputs])
            data = {
                'label': all_labels,
                'scores': all_scores,
                'pred': all_preds,
                # 'feat': all_feat
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

    def on_load_checkpoint(self, checkpoint) -> None:
        checkpoint['hyper_parameters']['cfg'].defrost()
        checkpoint['hyper_parameters']['cfg'].DATAS.ROOT_CELEB = '/home/og/home/lqx/datasets/Celeb-DF/Celeb-DF-v2'
        # checkpoint['hyper_parameters']['cfg'].MODEL.FREQ_DEPTH = 4
        # checkpoint['hyper_parameters']['cfg'].MODEL.FREQ_CHANNEL = 768
        checkpoint['hyper_parameters']['cfg'].DATAS.SOURCE_QUALITY = 'c40'
        checkpoint['hyper_parameters']['cfg'].DATAS.TARGET_QUALITY = 'c40'
        return checkpoint
