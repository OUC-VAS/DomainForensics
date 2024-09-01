from .base_config import ArgsParser


class ResNetParser(ArgsParser):
    def parse_dataloader_args(self, nprocs):
        batch_size = int(self.conf.data.dataloader.batch_size / nprocs)
        return {
            'source_path': self.args.source_path,
            'target_path': self.args.target_path,
            'evaluation_path': self.args.target_path,
            'batch_size': batch_size,
            'num_workers': self.conf.data.dataloader.data_workers,
            'use_balance': self.conf.data.dataloader.class_balance,
            'with_masked': self.conf.data.dataset.with_masked,
        }

    def parse_model_params(self):
        return {
            'base_model': self.conf.model.base_model,
            'proj_dim': self.conf.model.proj_dim
        }

    def parse_hyper_params(self):
        share = self.conf.data.dataset.n_share
        source_private = self.conf.data.dataset.n_source_private
        num_classes = share + source_private
        return {
            'share_classes': self.conf.data.dataset.n_share,
            'source_private_classes': self.conf.data.dataset.n_source_private,
            'num_classes': num_classes,
            'contrastive_weight': self.conf.train.contrastive_weight,
            'std_weight': self.conf.train.std_weight,
            'lr': self.conf.train.lr,
            'lr_proj': self.conf.train.lr_proj,
            'sgd_mom': self.conf.train.sgd_momentum,
            'weight_decay': self.conf.train.weight_decay,
            'steps': self.conf.train.min_step
        }