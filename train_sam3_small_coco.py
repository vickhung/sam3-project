#!/usr/bin/env python3
"""
Train SAM3-Small V2 on COCO dataset.

This script follows the original Meta SAM3 training pattern exactly.
Usage:
    # With torchrun (recommended):
    torchrun --nproc_per_node=1 train_sam3_small_coco.py -c training/configs/sam3_small_coco.yaml
    
    # Or single GPU without torchrun:
    python train_sam3_small_coco.py -c training/configs/sam3_small_coco.yaml --use-cluster 0 --num-gpus 1
"""

import os
import sys
import logging
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variable for full Hydra errors
os.environ["HYDRA_FULL_ERROR"] = "1"

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from iopath.common.file_io import g_pathmgr

from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process - matches original Meta pattern"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    # Instantiate trainer with _recursive_=False as in original
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    """Single node runner - matches original Meta pattern"""
    import torch.multiprocessing as mp
    
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch = __import__('torch')
    torch.multiprocessing.set_start_method("spawn")  # CUDA runtime does not support `fork`
    
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def main(args) -> None:
    """Main function - matches original Meta pattern"""
    import random
    import torch
    
    # Config path - use config_dir approach since we're not using config_module
    config_dir = os.path.join(os.path.dirname(__file__), "training", "configs")
    config_name = args.config.replace(".yaml", "").replace("training/configs/", "").replace("configs/", "")
    
    # Initialize hydra and load config
    with initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam3_logs", config_name
        )
    
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    experiment_log_dir = cfg.launcher.experiment_log_dir
    print(f"Experiment Log Dir:\n{experiment_log_dir}")

    # Prioritize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    
    if submitit_conf.use_cluster:
        # Cluster execution not implemented in this simplified version
        raise NotImplementedError("Cluster execution not supported in this script. Use original train.py for cluster support.")
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. training/configs/sam3_small_coco.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    main(args)
