from yacs.config import CfgNode
# :TODO complete config variables

DEFAULT_CONFIGS = CfgNode()

# data config
DEFAULT_CONFIGS.DATAS = CfgNode()
DEFAULT_CONFIGS.DATAS.ROOT = '/home/og/home/lqx/datasets/FF++'
DEFAULT_CONFIGS.DATAS.ROOT_CELEB = '/home/og/home/lqx/datasets/Celeb-DF/Celeb-DF-v2'
DEFAULT_CONFIGS.DATAS.ROOT_STYLEGAN = '/home/og/home/lqx/datasets/stylegan'
DEFAULT_CONFIGS.DATAS.SOURCE = "Deepfakes"
DEFAULT_CONFIGS.DATAS.SOURCE_QUALITY = "c40"
DEFAULT_CONFIGS.DATAS.TARGET = ["Face2Face"]
DEFAULT_CONFIGS.DATAS.TARGET_QUALITY = "c23"
DEFAULT_CONFIGS.DATAS.EXPAND_RATIO = 1.3
DEFAULT_CONFIGS.DATAS.WITH_FREQUENCY = False
DEFAULT_CONFIGS.DATAS.FREQ_MEAN_STD_PATH = ''
DEFAULT_CONFIGS.DATAS.DATA_RATIO = 1.0

# model config
DEFAULT_CONFIGS.MODEL = CfgNode()
DEFAULT_CONFIGS.MODEL.NAME = 'xception'
DEFAULT_CONFIGS.MODEL.NUM_CLASSES = 2
DEFAULT_CONFIGS.MODEL.CLASSIFICATION_FEATURE = 2048
DEFAULT_CONFIGS.MODEL.QUEUE_SIZE = 3200
DEFAULT_CONFIGS.MODEL.LOCAL_ALIGNMENT = False
DEFAULT_CONFIGS.MODEL.VIT_OUT_BLOCKS = [0, 3, 7, 11]
DEFAULT_CONFIGS.MODEL.FREQ_DEPTH = 4
DEFAULT_CONFIGS.MODEL.FREQ_CHANNEL = 768
# # discriminator
DEFAULT_CONFIGS.MODEL.DISCRIMINATOR_MIDDIM = 512


# augmentation cofig
DEFAULT_CONFIGS.AUGS = CfgNode()
DEFAULT_CONFIGS.AUGS.MEAN = (0.5, 0.5, 0.5)
DEFAULT_CONFIGS.AUGS.STD = (0.5, 0.5, 0.5)
DEFAULT_CONFIGS.AUGS.RESIZE_SIZE = (256, 256)
DEFAULT_CONFIGS.AUGS.CROP_SIZE = (224, 224)
DEFAULT_CONFIGS.AUGS.FLIP_PROB = 0.5
DEFAULT_CONFIGS.AUGS.CONTRAST_PROB = 0.5
DEFAULT_CONFIGS.AUGS.ROTATAION = 15
DEFAULT_CONFIGS.AUGS.CONTRASTIVE_ENABLE = False

# Training config
DEFAULT_CONFIGS.TRAINING = CfgNode()
DEFAULT_CONFIGS.TRAINING.LOG_PATH = '/path/to/logs'
DEFAULT_CONFIGS.TRAINING.BATCH_SIZE = 32
DEFAULT_CONFIGS.TRAINING.NUM_WORKERS = 4
DEFAULT_CONFIGS.TRAINING.MAX_EPOCHS = 10
DEFAULT_CONFIGS.TRAINING.MAX_STEPS = 20000
DEFAULT_CONFIGS.TRAINING.LR = 0.002
DEFAULT_CONFIGS.TRAINING.SCHEDULER = 'cosine'

DEFAULT_CONFIGS.DOMAIN_FINETUNING = CfgNode()
DEFAULT_CONFIGS.DOMAIN_FINETUNING.ENABLE = False
DEFAULT_CONFIGS.DOMAIN_FINETUNING.LR = 0.001
DEFAULT_CONFIGS.DOMAIN_FINETUNING.EPOCH = 10
DEFAULT_CONFIGS.DOMAIN_FINETUNING.DISCRIMINATOR = False

# Testing config
DEFAULT_CONFIGS.TESTING = CfgNode()
DEFAULT_CONFIGS.TESTING.TESTSETS = ["Face2Face"]
DEFAULT_CONFIGS.TESTING.BATCH_SIZE = 1
DEFAULT_CONFIGS.TESTING.NUM_WORKERS = 2
DEFAULT_CONFIGS.TESTING.MODEL_WEIGHT = '/path/to/weights'
DEFAULT_CONFIGS.TESTING.IMAGENUM_PER_VID = 8

DEFAULT_CONFIGS.LOG = CfgNode()
DEFAULT_CONFIGS.LOG.ROOT = '/home/og/home/lqx/conlogs'

def get_config():
    return DEFAULT_CONFIGS