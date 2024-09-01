import os.path

from models.base_banet_sbi import BaseFaceAdaptation
from datasets.loaders import FaceAdaptationLoaders
from trainer.ff_trainer import TrainerManager
from configs.defaults import get_config
import argparse


def update_finetune_cfg(cfg, gpus):
    cfg.defrost()
    cfg.LOG.ROOT = cfg.LOG.ROOT.replace('final', 'finetune')
    cfg.DOMAIN_FINETUNING.ENABLE = True
    cfg.TRAINING.BATCH_SIZE = 24
    cfg.TRAINING.BATCH_SIZE = cfg.TRAINING.BATCH_SIZE // len(gpus)
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/yamls/final.yaml', help='config file path')
    parser.add_argument('--project', default='final_sbi', help='config file path')
    parser.add_argument('--method', default='Deepfakes', help='config file path')
    parser.add_argument('--targetmethod', default='Deepfakes', help='config file path')
    parser.add_argument('--ratio', default=1.0, type=float)
    parser.add_argument('--freqdepth', default=4, type=int)
    parser.add_argument('--freqchannel', default=768, type=int)
    args = parser.parse_args()

    cfg = get_config()
    cfg.merge_from_file(args.config)
    cfg.LOG.ROOT = os.path.join(cfg.LOG.ROOT, args.project)

    gpus = [4,5]
    cfg.TRAINING.BATCH_SIZE = cfg.TRAINING.BATCH_SIZE // len(gpus)
    cfg.DATAS.SOURCE = ['SBI']
    cfg.DATAS.TARGET = ['FaceSwap']
    cfg.DATAS.SOURCE_QUALITY = 'c23'
    cfg.DATAS.TARGET_QUALITY = 'c23'
    cfg.DATAS.DATA_RATIO = args.ratio
    cfg.MODEL.FREQ_DEPTH = 4
    cfg.MODEL.FREQ_CHANNEL = 768
    cfg.TRAINING.MAX_EPOCHS = 20
    ratio_suffix = str(args.ratio).replace('.', '_')
    model_name = cfg.DATAS.SOURCE[0] + '_2_' + cfg.DATAS.TARGET[0]+ratio_suffix+'.ckpt'
    print("Save Model Names is : ", model_name)

    model_save_path = os.path.join(cfg.LOG.ROOT, 'save_models', model_name)

    ffloaders = FaceAdaptationLoaders(cfg=cfg, quality='c23', source_only=False)
    model = BaseFaceAdaptation(cfg)
    trainer = TrainerManager(gpus=gpus, cfg=cfg, save_path=model_save_path)

    trainer.multi_gpu_training(model, ffloaders)
    # finetune
    cfg = update_finetune_cfg(cfg, gpus)
    model.merge_from_other_cfg(cfg)
    trainer.multi_gpu_finetune(model, ffloaders)

