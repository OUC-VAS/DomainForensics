import os
import argparse
import easydict
import yaml


def initialize_args():
    parser = argparse.ArgumentParser(description='Pytorch DA',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--source_path', type=str, default='./util/source_list.txt', metavar='B',
                        help='path to source list')
    parser.add_argument('--target_path', type=str, default='./util/target_list.txt', metavar='B',
                        help='path to target list')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--exp_name', type=str, default='office_close', help='/path/to/config/file')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

    args = parser.parse_args()
    return args


class ArgsParser(object):
    def __init__(self, args):
        super(ArgsParser, self).__init__()
        self.args = args
        self.conf = easydict.EasyDict(yaml.load(open(args.config), Loader=yaml.FullLoader))

    def parse_dataloader_args(self, nprocs):
        share = self.conf.data.dataset.n_share
        source_private = self.conf.data.dataset.n_source_private
        num_classes = share + source_private
        batch_size = int(self.conf.data.dataloader.batch_size / nprocs)
        return {
            'share_classes': self.conf.data.dataset.n_share,
            'source_private_classes': self.conf.data.dataset.n_source_private,
            'num_classes': num_classes,
            'source_path': self.args.source_path,
            'target_path': self.args.target_path,
            'evaluation_path': self.args.target_path,
            'batch_size': batch_size,
            'num_workers': self.conf.data.dataloader.data_workers,
            'use_balance': self.conf.data.dataloader.class_balance,
            'with_masked': self.conf.data.dataset.with_masked,
        }

    def parse_log_params(self, log_file_name):
        domain_name = self.args.source_path.split("_")[1] + "2" + self.args.target_path.split("_")[1]
        dir_path = os.path.join("record", self.args.exp_name, self.args.config.replace(".yaml", ""))
        full_path = os.path.join(dir_path, domain_name)

        return {
            'log_dir_name': domain_name,
            'log_dir_path': dir_path,
            'full_log_path': full_path,
            'log_file_name': log_file_name
        }

    def parse_model_params(self):
        return {
            'base_model': self.conf.model.base_model,
            'latent_dim': self.conf.model.latent_dim,
            'hidden_dim': self.conf.model.hidden_dim,
            'temp': self.conf.model.temp
        }

    def parse_hyper_params(self):
        return {
            'ent_n_w': self.conf.entropy_enhancement.negative_weight,
            'ent_p_w': self.conf.entropy_enhancement.positive_weight,
            'ent_n_thresh': self.conf.entropy_enhancement.negative_thresh,
            'ent_p_thresh': self.conf.entropy_enhancement.positive_thresh,
            'gene_weight': self.conf.train.gene_weight,
            'recon_weight': self.conf.train.recon_weight,
            'ent_weight': self.conf.train.entropy_weight,
            'contrastive_weight': self.conf.train.contrastive_weight,
            'test_threshold': self.conf.test.test_threshold
        }

    def parse_train_params(self):
        return {
            'init_lr': self.conf.train.lr,
            'min_step': self.conf.train.min_step,
            'lr_backbone': self.conf.train.multi,
            'wdecay_backbone': self.conf.train.weight_decay,
            'sgd_momentum': self.conf.train.sgd_momentum,
            'lr_lafea': self.conf.train.lr_lafea,
            'lr_proj': self.conf.train.lr_proj,
            'log_interval': self.conf.train.log_interval,
            'test_interval': self.conf.test.test_interval
        }

