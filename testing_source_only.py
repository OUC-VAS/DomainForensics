# from models.base import BaseFaceAdaptation
# from models.base_so import BaseFaceAdaptation
# from models.base_ssrt import BaseFaceAdaptation
from models.base_vit_so import BaseFaceAdaptation
from datasets.loaders import FaceAdaptationLoaders
from trainer.ff_trainer import TrainerManager
from configs.defaults import get_config
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/yamls/vitbase_so.yaml', help='config file path')
    args = parser.parse_args()

    cfg = get_config()
    cfg.merge_from_file(args.config)
    quality = 'c23'

    # cfg.TESTING.MODEL_WEIGHT = '/home/og/home/lqx/conlogs/vit_so/lightning_logs/version_30/checkpoints/epoch=39-step=14360.ckpt'
    # cfg.TESTING.MODEL_WEIGHT = '/home/og/home/lqx/conlogs/vit_so/save_models/Deepfakes_'+quality+'_so.ckpt'
    cfg.TESTING.MODEL_WEIGHT = '/home/og/home/lqx/conlogs/vit_so_review/lightning_logs/version_4/checkpoints/epoch=9-step=8990.ckpt'
    # cfg.TESTING.MODEL_WEIGHT = '/home/og/home/lqx/conlogs/final/save_models/vit_so/NeuralTextures_' + quality + '_so.ckpt'

    # methods = ["Deepfakes", "Face2Face", 'NeuralTextures', "FaceSwap"]
    # methods = ["Deepfakes", "Face2Face", "NeuralTextures"]
    # methods = ['FaceSwap']
    methods = ['CelebDF']
    
    
    # print('*'*50, '    ', method_list , '    ', '*'*50)
    cfg.defrost()
    cfg.TESTING.TESTSETS = methods
    print(cfg.TESTING.TESTSETS)
    cfg.freeze()
    ffloaders = FaceAdaptationLoaders(cfg, quality='c23') # quality
    model = BaseFaceAdaptation.load_from_checkpoint(cfg.TESTING.MODEL_WEIGHT)
    trainer = TrainerManager(gpus=[2])
    trainer.single_gpu_test(model=model, data_module=ffloaders)
    # model_eval(cfg, gpus=[5], quality='c40')
    
    # for m in methods:
    #     cfg.defrost()
    #     cfg.TESTING.TESTSETS = [m]
    #     cfg.freeze()

    #     ffloaders = FaceAdaptationLoaders(cfg, quality='c23') # quality
    #     model = BaseFaceAdaptation.load_from_checkpoint(cfg.TESTING.MODEL_WEIGHT)
    #     trainer = TrainerManager(gpus=[6])
    #     trainer.single_gpu_test(model=model, data_module=ffloaders)

