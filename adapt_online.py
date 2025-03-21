import os
import time
import argparse
import numpy as np
import random
import torch
# import pipelines.tent as tent

import models
from models import MinkUNet18_HEADS, MinkUNet18_MCMC
from utils.config import get_config
from utils.collation import CollateSeparated, CollateFN
from utils.dataset_online import get_online_dataset
# from utils.online_logger import OnlineWandbLogger, OnlineCSVLogger
from utils.pseudo import PseudoLabel
from pipelines import OneDomainAdaptation, OnlineTrainer

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(1234)

torch.cuda.set_device(1)


parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/deva/nuscenes_sequence.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--split_size",
                    default=4071,
                    type=int,
                    help="Num frames per sub sequence (SemanticKITTI only)")
parser.add_argument("--drop_prob",
                    default=None,
                    type=float,
                    help="Dropout prob MCMC")
parser.add_argument("--save_predictions",
                    default=False,
                    action='store_true')
parser.add_argument("--use_global",
                    default=False,
                    action='store_true')

AUG_DICT = None


def get_mini_config(main_c):
    return dict(time_window=main_c.dataset.max_time_window,
                mcmc_it=main_c.pipeline.num_mc_iterations,
                metric=main_c.pipeline.metric,
                cbst_p=main_c.pipeline.top_p,
                th_pseudo=main_c.pipeline.th_pseudo,
                top_class=main_c.pipeline.top_class,
                propagation_size=main_c.pipeline.propagation_size,
                drop_prob=main_c.model.drop_prob)


def train(config, split_size=4071, save_preds=False):

    mapping_path = config.dataset.mapping_path

    eval_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                      dataset_path=config.dataset.dataset_path,
                                      voxel_size=config.dataset.voxel_size,
                                      augment_data=config.dataset.augment_data,
                                      max_time_wdw=config.dataset.max_time_window,
                                      version=config.dataset.version,
                                      sub_num=config.dataset.num_pts,
                                      ignore_label=config.dataset.ignore_label,
                                      split_size=split_size,
                                      mapping_path=mapping_path,
                                      num_classes=config.model.out_classes)

    adapt_dataset = get_online_dataset(dataset_name=config.dataset.name,
                                       dataset_path=config.dataset.dataset_path,
                                       voxel_size=config.dataset.voxel_size,
                                       augment_data=config.dataset.augment_data,
                                       max_time_wdw=config.dataset.max_time_window,
                                       version=config.dataset.version,
                                       sub_num=config.dataset.num_pts,
                                       ignore_label=config.dataset.ignore_label,
                                       split_size=split_size,
                                       mapping_path=mapping_path,
                                       num_classes=config.model.out_classes)

    Model = getattr(models, config.model.name)
    source_model = Model(config.model.in_feat_size, config.model.out_classes)
    ema_model = Model(config.model.in_feat_size, config.model.out_classes)

    module = OneDomainAdaptation(eval_dataset=eval_dataset,
                                 adapt_dataset=adapt_dataset,
                                 model = source_model,
                                 num_classes=config.model.out_classes,
                                 source_model=ema_model,
                                 epsilon=config.pipeline.eps,
                                 seg_beta=config.pipeline.segmentation_beta,
                                 adaptation_batch_size=config.pipeline.dataloader.adaptation_batch_size,
                                 stream_batch_size=config.pipeline.dataloader.stream_batch_size,
                                 clear_cache_int=config.pipeline.trainer.clear_cache_int,
                                 train_num_workers=config.pipeline.dataloader.num_workers,
                                 val_num_workers=config.pipeline.dataloader.num_workers,
                                 use_random_wdw=config.pipeline.random_time_window,
                                 freeze_list=config.pipeline.freeze_list,
                                 num_mc_iterations=config.pipeline.num_mc_iterations)

    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())
    run_name = run_time

    mini_configs = get_mini_config(config)

    for k, v in mini_configs.items():
        run_name += f'_{str(k)}:{str(v)}'

    save_dir = os.path.join(config.pipeline.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # wandb_logger = OnlineWandbLogger(project=config.pipeline.wandb.project_name,
    #                                  entity=config.pipeline.wandb.entity_name,
    #                                  name=run_name,
    #                                  offline=config.pipeline.wandb.offline,
    #                                  config=mini_configs)

    # csv_logger = OnlineCSVLogger(save_dir=save_dir,
    #                              version='logs')

    # loggers = [wandb_logger, csv_logger]

    try:
        is_spatiotemporal = config.pipeline.is_spatiotemporal
    except:
        is_spatiotemporal =False

    trainer = OnlineTrainer(pipeline=module,
                            collate_fn_eval=CollateFN(),
                            collate_fn_adapt=CollateFN(),
                            device=config.pipeline.gpu,
                            default_root_dir=config.pipeline.save_dir,
                            weights_save_path=os.path.join(save_dir, 'checkpoints'),
                            save_checkpoint_every=config.pipeline.trainer.save_checkpoint_every,
                            source_checkpoint=config.pipeline.source_model,
                            student_checkpoint=config.pipeline.student_model,
                            use_mcmc=config.pipeline.use_mcmc,
                            sub_epochs=config.pipeline.sub_epoch,
                            save_predictions=save_preds,
                            corrupt_type = config.dataset.corrupt_type,
                            level = config.dataset.level)

    trainer.adapt_double()


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    train(config, split_size=args.split_size, save_preds=args.save_predictions)
