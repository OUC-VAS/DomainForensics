import os.path

from models.base_vit_so import BaseFaceAdaptation
from datasets.loaders import FaceAdaptationLoaders
from trainer.ff_trainer import TrainerManager
from configs.defaults import get_config
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/yamls/vitbase_so.yaml', help='config file path')
    parser.add_argument('--project', default='vit_so_review', help='config file path')
    parser.add_argument('--method', default='Deepfakes', help='config file path')
    parser.add_argument('--quality', default='c23', help='config file path')
    args = parser.parse_args()

    cfg = get_config()
    cfg.merge_from_file(args.config)
    cfg.LOG.ROOT = os.path.join(cfg.LOG.ROOT, args.project)

    cfg.TRAINING.MAX_EPOCHS = 20

    gpus = [6]

    cfg.TRAINING.BATCH_SIZE = cfg.TRAINING.BATCH_SIZE // len(gpus)
    # cfg.DATAS.SOURCE = args.method
    # cfg.DATAS.SOURCE = ['SBI']
    cfg.DATAS.SOURCE = ["Deepfakes", "Face2Face", 'NeuralTextures', "FaceSwap"]
    # cfg.DATAS.SOURCE = ["Deepfakes"]
    cfg.DATAS.TARGET = ['Deepfakes']
    cfg.DATAS.SOURCE_QUALITY = 'c23'
    cfg.DATAS.TARGET_QUALITY = 'c23'
    model_name = cfg.DATAS.SOURCE[0] + '_' + str(args.quality) + '_so.ckpt'
    model_save_path = os.path.join(cfg.LOG.ROOT, 'save_models', model_name)
    ffloaders = FaceAdaptationLoaders(cfg=cfg, quality=args.quality, source_only=True)
    model = BaseFaceAdaptation(cfg)
    trainer = TrainerManager(gpus=gpus, cfg=cfg, save_path=model_save_path)

    trainer.single_gpu_training(model, ffloaders)
    # resume_ckpt = '/home/og/home/lqx/conlogs/vit_so/lightning_logs/version_22/checkpoints/epoch=4-step=1795.ckpt'
    # trainer.multi_gpu_training(model, ffloaders) # , resume_ckpt=resume_ckpt

    # trainer.single_gpu_eval()