# Copyright NavInfo Europe 2023. Adapted from Packnet-sfm Toyota Research Institute. 

import argparse

from mtsfmlearner.models.model_wrapper import ModelWrapper
from mtsfmlearner.models.model_checkpoint import ModelCheckpoint
from mtsfmlearner.trainers.horovod_trainer import HorovodTrainer
from mtsfmlearner.utils.config import parse_train_file
from mtsfmlearner.utils.load import set_debug, filter_args_create
from mtsfmlearner.utils.horovod import hvd_init, rank
from mtsfmlearner.loggers import WandbLogger


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='MT-SfMLearner training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = None if config.wandb.dry_run or rank() > 0 \
        else filter_args_create(WandbLogger, config.wandb)

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    train(args.file)
