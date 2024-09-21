from models.base_banet import BaseFaceAdaptation
from datasets.loaders import FaceAdaptationLoaders
from trainer.ff_trainer import TrainerManager
from configs.defaults import get_config
import argparse


def model_eval(cfg, gpus=[0], quality='c40'):
    ffloaders = FaceAdaptationLoaders(cfg, quality=quality)
    model = BaseFaceAdaptation.load_from_checkpoint(cfg.TESTING.MODEL_WEIGHT)
    trainer = TrainerManager(gpus=gpus)
    trainer.single_gpu_test(model=model, data_module=ffloaders)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/yamls/final.yaml', help='config file path')
    args = parser.parse_args()

    cfg = get_config()
    cfg.merge_from_file(args.config)

    cfg.MODEL.FREQ_DEPTH = 4
    cfg.MODEL.FREQ_CHANNEL = 768
    # e.g. cfg.TESTING.MODEL_WEIGHT = 'version_1/checkpoints/epoch=14-step=10770.ckpt'
    cfg.TESTING.MODEL_WEIGHT = 'path_to_model'


    method_list = ["Deepfakes"]
    method_list = ["Face2Face"]


    print('*'*50, '    ', method_list , '    ', '*'*50)
    cfg.defrost()
    cfg.TESTING.TESTSETS = method_list
    print(cfg.TESTING.TESTSETS)
    cfg.freeze()
    model_eval(cfg, gpus=[0], quality='c40')




