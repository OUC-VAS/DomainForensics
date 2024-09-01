import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.ssrt_con import SSRT
from models.components.xception import return_pytorch04_xception
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from models.components.effcientnet import get_efficientnet
from models.transfer_base import TransferModel

class BaseFaceAdaptation(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseFaceAdaptation, self).__init__()

        self.save_hyperparameters()
        feat_dim = self.hparams.cfg.MODEL.CLASSIFICATION_FEATURE
        self.model = TransferModel(cfg=cfg, source_only=True)

        self.backbone = self.model.net

        self.gobal_i = 0
        self.automatic_optimization = False
        self.local_alignment = self.hparams.cfg.MODEL.LOCAL_ALIGNMENT
        self.return_name = True

    def forward(self, x):
        _, out, _ = self.backbone(x)
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        opt.zero_grad()

        self.gobal_i += 1
        batch_s = batch['source']

        # source
        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            s_x, s_ycbcr, s_y = batch_s
        else:
            s_x, s_y = batch_s
            # s_x_real, s_x_fake = batch_s
            # s_x = torch.cat([s_x_real, s_x_fake], dim=0)
            # s_y = torch.cat([torch.zeros(s_x_real.size()[0]), torch.ones(s_x_real.size()[0])]).to(s_x.device).long()

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            total_loss = self.model.get_loss(inputs_source=s_x, labels_source=s_y, inputs_s_ycbcr=s_ycbcr)
        else:
            total_loss = self.model.get_loss(inputs_source=s_x, labels_source=s_y)

        cls_loss = total_loss.detach()
        self.manual_backward(total_loss)
        opt.step()
        opt.zero_grad()
        # single scheduler
        sch = self.lr_schedulers()
        sch.step()

        self.log('ce_loss', float(cls_loss.cpu()), on_step=True)

        return total_loss

    def configure_optimizers(self):
        params = self.model.get_parameter_list()

        optim = torch.optim.SGD(
            params, lr=self.hparams.cfg.TRAINING.LR, momentum=0.9, nesterov=True, weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: self.hparams.cfg.TRAINING.LR * (1. + 0.001 * float(x)) ** (-0.75))

        return [optim], [scheduler]

    def _model_predict(self, batch, batch_idx, mode='train'):

        if self.hparams.cfg.DATAS.WITH_FREQUENCY:
            if self.return_name:
                x, freq_x, y, vid_num = batch
            else:
                x, freq_x, y = batch
        else:
            if self.return_name:
                x, y = batch
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
                    'feat': pred_features[0, 0].unsqueeze(0),
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

    def _eval_results(self, outputs, save_pth=False):
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
                'feat': all_feat,
            }
            torch.save(data, './outs/pred_so.pth')
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
        self.backbone.eval()
        return self._model_predict(batch, batch_idx, mode='test')

    def test_epoch_end(self, test_step_outputs):
        self._eval_results(test_step_outputs, save_pth=True)


