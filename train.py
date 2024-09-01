import os.path
import torch

from models.base_banet import BaseFaceAdaptation
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
    cfg.TESTING.MODEL_WEIGHT = ''
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/yamls/final.yaml', help='config file path')
    parser.add_argument('--project', default='tsne', help='config file path')
    parser.add_argument('--method', default='Deepfakes', help='config file path')
    parser.add_argument('--targetmethod', default='Deepfakes', help='config file path')
    parser.add_argument('--ratio', default=1.0, type=float)
    parser.add_argument('--freqdepth', default=4, type=int)
    parser.add_argument('--freqchannel', default=768, type=int)
    args = parser.parse_args()

    cfg = get_config()
    cfg.merge_from_file(args.config)
    cfg.LOG.ROOT = os.path.join(cfg.LOG.ROOT, args.project)

    gpus = [0,1]
    
    cfg.TRAINING.BATCH_SIZE = cfg.TRAINING.BATCH_SIZE // len(gpus)
    cfg.DATAS.SOURCE = ["Deepfakes"]
    cfg.DATAS.TARGET = ['Face2Face']
    if cfg.DATAS.TARGET == ['CelebDF']:
        cfg.TRAINING.MAX_EPOCHS = 10
    cfg.DATAS.SOURCE_QUALITY = 'c40'
    cfg.DATAS.TARGET_QUALITY = 'c40'
    cfg.DATAS.DATA_RATIO = args.ratio
    cfg.MODEL.FREQ_DEPTH = 4
    cfg.MODEL.FREQ_CHANNEL = 768
    cfg.TESTING.MODEL_WEIGHT = ''
    ratio_suffix = str(args.ratio).replace('.', '_')
    model_name = cfg.DATAS.SOURCE[0] + '_2_' + cfg.DATAS.TARGET[0]+ratio_suffix+'.ckpt'
    print("Save Model Names is : ", model_name)

    model_save_path = os.path.join(cfg.LOG.ROOT, 'save_models', model_name)

    ffloaders = FaceAdaptationLoaders(cfg=cfg, quality='c23', source_only=False, gpus=gpus)
    model = BaseFaceAdaptation(cfg)
    
    trainer = TrainerManager(gpus=gpus, cfg=cfg, save_path=model_save_path)

    # foward
    trainer.multi_gpu_training(model, ffloaders)

    cfg = update_finetune_cfg(cfg, gpus)
    # backward
    model.merge_from_other_cfg(cfg)
    trainer.multi_gpu_finetune(model, ffloaders)