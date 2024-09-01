import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.resnet import ResnetBase, ResClassifierMME
from models.components.discriminator import Discriminator

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from models.components.effcientnet import get_efficientnet
from models.components.xception import return_pytorch04_xception

def lambda_func(x):
    return (1. + 0.001 * float(x)) ** (-0.75) * 0.001


class BaseFaceAdaptation(pl.LightningModule):
    def __init__(self, cfg):
        super(BaseFaceAdaptation, self).__init__()

        self.save_hyperparameters()
        self.backbone = return_pytorch04_xception(pretrained=True)
        # classifier
        self.classifier = ResClassifierMME(num_classes=2, input_size=2048)
        self.gobal_i = 0
        self.automatic_optimization = False

    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)
        return out

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        self.gobal_i += 1
        batch_s = batch['source']

        # source
        s_x, s_y = batch_s
        s_feat = self.backbone(s_x)
        s_y_hat = self.classifier(s_feat)
        loss = F.cross_entropy(s_y_hat, s_y)

        loss = loss
        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.log('ce_loss', float(loss.detach().cpu()), on_step=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.classifier.parameters()}
        ],
            lr=0.0002, betas=(0.9, 0.999), eps=1e-8
        )

        return optim

    def _model_predict(self, batch, batch_idx):
        x, y = batch
        feat = self.backbone(x)
        y_hat = self.classifier(feat)
        y_hat = F.softmax(y_hat, dim=-1)
        entropy = -torch.sum(y_hat * torch.log(y_hat + 1e-5), 1)
        pred = torch.max(y_hat, dim=1)[1]
        return {
            'pred': pred,
            'label': y,
            'entropy': entropy,
            'feat': feat,
        }

    def _eval_results(self, outputs):
        # TODO: complete auc and other evaluation metric
        all_preds = torch.cat([_['pred'] for _ in outputs])
        all_labels = torch.cat([_['label'] for _ in outputs])
        all_ent = torch.cat([_['entropy'] for _ in outputs])
        all_feat = torch.cat([_['feat'] for _ in outputs])
        # acc = torch.sum(all_preds == all_labels) / len(all_labels)
        data = {
            'label': all_labels,
            'pred': all_preds,
            'entropy': all_ent,
            'feat': all_feat
        }
        torch.save(data, 'pred.pth')
        # print("label : ", all_labels)
        # print("pred : ", all_preds)
        try:
            acc = accuracy_score(all_labels.detach().cpu().numpy(), all_preds.detach().cpu().numpy())
            auc = roc_auc_score(all_labels.detach().cpu().numpy(), all_preds.detach().cpu().numpy())
            self.log("accuracy", float(acc)*100, on_epoch=True)
            self.log("auc", float(auc)*100, on_epoch=True)
        except:
            self.log("accuracy", 0.0, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self._model_predict(batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        self._eval_results(validation_step_outputs)

    def test_step(self, batch, batch_idx):
        return self._model_predict(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        self._eval_results(test_step_outputs)

