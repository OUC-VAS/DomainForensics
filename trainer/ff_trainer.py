import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from configs.defaults import CfgNode
import os


class TrainerManager():
    def __init__(self, gpus, cfg:CfgNode=None, save_path=None):
        super(TrainerManager, self).__init__()
        self.gpus = gpus
        self.cfg = cfg
        self.save_path = save_path

    def update_finetune_cfg(self, model):
        model.hparams.cfg.defrost()
        model.hparams.cfg.LOG.ROOT = model.hparams.cfg.LOG.ROOT.replace('final', 'finetune')
        model.hparams.cfg.DOMAIN_FINETUNING.ENABLE = True
        model.hparams.cfg.TRAINING.BATCH_SIZE = 24
        model.hparams.cfg.TRAINING.BATCH_SIZE = model.hparams.cfg.TRAINING.BATCH_SIZE // len(self.gpus)

        model.hparams.cfg.freeze()

    def single_gpu_training(self, model, data_module):
        max_epochs = self.cfg.TRAINING.MAX_EPOCHS
        self.cfg.freeze()
        trainer = Trainer(gpus=self.gpus, accelerator="gpu", max_epochs=max_epochs, log_every_n_steps=200,
                          default_root_dir=self.cfg.LOG.ROOT, logger=pl_loggers.CSVLogger(save_dir=self.cfg.LOG.ROOT),
                          check_val_every_n_epoch=5)
        trainer.fit(model, datamodule=data_module)
        trainer.validate(model=model, datamodule=data_module)

    def multi_gpu_training(self, model, data_module, resume_ckpt=None):
        max_epochs = self.cfg.TRAINING.MAX_EPOCHS
        self.cfg.freeze()
        ddp = DDPStrategy(process_group_backend="nccl")
        trainer = Trainer(devices=self.gpus, accelerator="gpu", strategy=ddp,
                          log_every_n_steps=200, max_epochs=max_epochs,
                          default_root_dir=self.cfg.LOG.ROOT, logger=pl_loggers.CSVLogger(save_dir=self.cfg.LOG.ROOT),
                          check_val_every_n_epoch=5)

        if resume_ckpt is None:
            trainer.fit(model, datamodule=data_module)
        else:
            trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
        trainer.save_checkpoint(self.save_path)

    def multi_gpu_finetune(self, model, data_module):
        ddp = DDPStrategy(process_group_backend="nccl")
        fine_tune_epoch = self.cfg.DOMAIN_FINETUNING.EPOCH
        finetune_trainer = Trainer(devices=self.gpus, accelerator="gpu", strategy=ddp,
                          log_every_n_steps=200, max_epochs=fine_tune_epoch,
                          default_root_dir=self.cfg.LOG.ROOT, logger=pl_loggers.CSVLogger(save_dir=self.cfg.LOG.ROOT),
                          check_val_every_n_epoch=5)
        finetune_trainer.fit(model, datamodule=data_module)
        finetune_trainer.save_checkpoint(self.save_path[:-5] + '_ft.ckpt')

    def single_gpu_eval(self, model, data_module):
        pass

    def single_gpu_test(self, model, data_module):
        trainer = Trainer(devices=self.gpus, accelerator="gpu", logger=False)# , callbacks=[RichProgressBar()]
        trainer.test(model, datamodule=data_module)
