from datasets.ffpp import FFDatasets, FFTestDataset
from datasets.celebdf import CelebDatasets
from datasets.stylegan import StyleGANDatasets
from datasets.ff_sbi import FFSBIDatasets
from datasets.dfdcp import DFDCPDatasets
from datasets.ffiw import FFIWDatasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets.trans import trans_wrapper, get_transfroms, get_contrastive_trans
from torch.utils.data.distributed import DistributedSampler


class FaceAdaptationLoaders(LightningDataModule):
    def __init__(self, cfg, quality='c40', source_only=False, gpus=[0]):
        super(FaceAdaptationLoaders, self).__init__()
        self.cfg = cfg
        self.quality = quality
        self.source_only = source_only
        self.gpus = gpus

    def train_dataloader(self):
        trans = get_transfroms(self.cfg, mode='train')
        if not self.source_only:
            source_quality = self.cfg.DATAS.SOURCE_QUALITY
            target_quality = self.cfg.DATAS.TARGET_QUALITY
            if self.cfg.DATAS.SOURCE[0] == 'SBI':
                source_dp = FFSBIDatasets(cfg=self.cfg, mode='train', quality=source_quality)
                batch_size_source = self.cfg.TRAINING.BATCH_SIZE // 2
            else:
                source_dp = FFDatasets(self.cfg, mode='train', method=self.cfg.DATAS.SOURCE, quality=source_quality,
                                       trans=trans, split_ratio=1.0)
                batch_size_source = self.cfg.TRAINING.BATCH_SIZE
            if self.cfg.DATAS.TARGET[0] == 'CelebDF':
                target_dp = CelebDatasets(cfg=self.cfg, mode='train', trans=trans, split_ratio=1.0)
                batch_size_source = 16
                batch_size_target = 16
                if self.cfg.DOMAIN_FINETUNING.ENABLE:
                    batch_size_source = 8
                    batch_size_target = 16
            elif self.cfg.DATAS.TARGET[0] == 'StyleGAN':
                target_dp = StyleGANDatasets(cfg=self.cfg, mode='train', trans=trans, split_ratio=1.0)
                batch_size_target = self.cfg.TRAINING.BATCH_SIZE
            elif self.cfg.DATAS.TARGET[0] == 'SBI':
                target_dp = FFSBIDatasets(cfg=self.cfg, mode='train', quality=target_quality)
                batch_size_target = self.cfg.TRAINING.BATCH_SIZE // 2
            elif self.cfg.DATAS.TARGET[0] == 'DFDCP':
                target_dp = DFDCPDatasets(self.cfg, mode='train', trans=trans)
                batch_size_target = self.cfg.TRAINING.BATCH_SIZE
            elif self.cfg.DATAS.TARGET[0] == 'FFIW':
                target_dp = FFIWDatasets(self.cfg, mode='train', trans=trans)
                batch_size_target = self.cfg.TRAINING.BATCH_SIZE
            else:
                target_dp = FFDatasets(self.cfg, mode='train', method=self.cfg.DATAS.TARGET, quality=target_quality,
                                       trans=trans, split_ratio=self.cfg.DATAS.DATA_RATIO) # self.cfg.DATAS.DATA_RATIO
                batch_size_target = self.cfg.TRAINING.BATCH_SIZE
            
            if len(self.gpus) > 1:
                source_sampler = None
                target_sampler = None
            else:
                source_sampler = None
                target_sampler = None

            source_loader = DataLoader(source_dp, num_workers=self.cfg.TRAINING.NUM_WORKERS//2,
                                       shuffle=(source_sampler is None), sampler=source_sampler, batch_size=batch_size_source, drop_last=True)
            target_loader = DataLoader(target_dp, num_workers=self.cfg.TRAINING.NUM_WORKERS//2,
                                       shuffle=(target_sampler is None), sampler=target_sampler, batch_size=batch_size_target, drop_last=True)
            return {
                'source': source_loader,
                'target': target_loader
            }
        else:
            if self.cfg.DATAS.SOURCE[0] == 'CelebDF':
                source_dp = CelebDatasets(cfg=self.cfg, mode='train', trans=trans, split_ratio=1.0)
                batch_size_source = self.cfg.TRAINING.BATCH_SIZE
            elif self.cfg.DATAS.SOURCE[0] == 'StyleGAN':
                source_dp = StyleGANDatasets(cfg=self.cfg, mode='train', trans=trans, split_ratio=1.0)
                batch_size_source = self.cfg.TRAINING.BATCH_SIZE
            elif self.cfg.DATAS.SOURCE[0] == 'SBI':
                source_dp = FFSBIDatasets(cfg=self.cfg, mode='train', quality=self.quality)
                batch_size_source = self.cfg.TRAINING.BATCH_SIZE // 2
            else:
                source_dp = FFDatasets(self.cfg, mode='train', method=self.cfg.DATAS.SOURCE, quality=self.quality,
                                       trans=trans)
                batch_size_source = self.cfg.TRAINING.BATCH_SIZE

            source_loader = DataLoader(source_dp, num_workers=self.cfg.TRAINING.NUM_WORKERS,
                                       shuffle=True, batch_size=batch_size_source, drop_last=True)
            return {
                'source': source_loader
            }

    def val_dataloader(self):
        trans = get_transfroms(self.cfg, mode='val')
        if self.cfg.DATAS.TARGET[0] == 'CelebDF':
            target_dp = CelebDatasets(cfg=self.cfg, mode='train', trans=trans, split_ratio=0.05)
        elif self.cfg.DATAS.TARGET[0] == 'StyleGAN':
            target_dp = StyleGANDatasets(cfg=self.cfg, mode='test', trans=trans, split_ratio=1.0)
        elif self.cfg.DATAS.TARGET[0] == 'SBI':
            target_dp = FFDatasets(self.cfg, mode='val', method=self.cfg.DATAS.SOURCE, quality=self.quality,
                                   trans=trans)
        elif self.cfg.DATAS.TARGET[0] == 'DFDCP':
            target_dp = DFDCPDatasets(self.cfg, mode='test', trans=trans)
        elif self.cfg.DATAS.TARGET[0] == 'FFIW':
            target_dp = FFIWDatasets(self.cfg, mode='test', trans=trans)
        else:
            target_dp = FFDatasets(self.cfg, mode='val', method=self.cfg.DATAS.TARGET, quality=self.quality, trans=trans)
        target_loader = DataLoader(target_dp, num_workers=4,
                                   shuffle=False, batch_size=1)
        return target_loader

    def test_dataloader(self):
        trans = get_transfroms(self.cfg, mode='test')
        if self.cfg.TESTING.TESTSETS[0] == 'CelebDF':
            source_dp = CelebDatasets(cfg=self.cfg, mode='test', trans=trans)
            test_batch_size = self.cfg.TESTING.BATCH_SIZE
        elif self.cfg.TESTING.TESTSETS[0] == 'StyleGAN':
            source_dp = StyleGANDatasets(cfg=self.cfg, mode='test', trans=trans, split_ratio=1.0, return_name=True)
            test_batch_size = 16
        elif self.cfg.TESTING.TESTSETS[0] == 'DFDCP':
            source_dp = DFDCPDatasets(self.cfg, mode='test', trans=trans)
            test_batch_size = self.cfg.TESTING.BATCH_SIZE
        elif self.cfg.TESTING.TESTSETS[0] == 'FFIW':
            source_dp = FFIWDatasets(self.cfg, mode='test', trans=trans)
            test_batch_size = self.cfg.TESTING.BATCH_SIZE
        else:
            source_dp = FFTestDataset(self.cfg, mode='test', method=self.cfg.TESTING.TESTSETS, quality=self.quality, trans=trans)
            test_batch_size = self.cfg.TESTING.BATCH_SIZE
        source_loader = DataLoader(source_dp, num_workers=self.cfg.TESTING.NUM_WORKERS,
                                   shuffle=False, batch_size=test_batch_size)
        return source_loader


