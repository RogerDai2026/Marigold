# Author: Bingxin Ke
# Last modified: 2024-03-12

import logging
import os
import sys
import wandb
from tabulate import tabulate
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

def get_visualization_image(image_rgb, depth_GT, pred_depth):
    # create a single array with 3 channels
    # height should be common, width should be 5 times
    # depth_GT: [H, W, 3]
    h = depth_GT.shape[0]
    w = depth_GT.shape[1]
    vis_image = np.zeros((h, 5 * w, 3), dtype=np.uint8)

    def convert_to_suitable_format(img, h, w):
        if (img.shape[0] != h) or (img.shape[1] != w):
            img = img[:h, :w]
        if (len(img.shape) == 2):
            img = img[:,:,None].repeat(3, axis=-1)

        if img.dtype == np.float32:
            if img.max() > 10:
                img = np.clip(img, a_min=0, a_max=250.0)
                img = img.astype(np.uint8)
            else:
                img = (img * 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = (img/256).astype(np.uint8)
        return img
    
    def convert_to_inverse_log_magma(img, h, w, amax, amin=None):
        img_temp = img[:h, :w]
        img_temp = np.clip(img_temp, a_min = amin, a_max=amax)
        img_temp = np.log(1/img_temp)
        img_temp = (img_temp - img_temp.min())/np.abs(img_temp.max() - img_temp.min())
        img_temp = plt.cm.magma(img_temp)[:,:,:3]
        img_magma = (img_temp * 255).astype(np.uint8)
        return img_magma
    
    img_rgb = convert_to_suitable_format(image_rgb, h, w)
    img_depth_GT = convert_to_suitable_format(depth_GT, h, w)
    img_depth_pred = convert_to_suitable_format(pred_depth, h, w)
    img_depth_GT_magma = convert_to_inverse_log_magma(depth_GT, h, w, amin=5, amax=250)
    img_depth_magma = convert_to_inverse_log_magma(pred_depth, h, w, amin=5, amax=250)

    vis_image[:, 0:w] = img_rgb
    vis_image[:, w:2*w] = img_depth_GT
    vis_image[:, 2*w:3*w] = img_depth_pred
    vis_image[:, 3*w:4*w] = img_depth_GT_magma
    vis_image[:, 4*w:5*w] = img_depth_magma

    return vis_image
    


def config_logging(cfg_logging, out_dir=None):
    file_level = cfg_logging.get("file_level", 10)
    console_level = cfg_logging.get("console_level", 10)

    log_formatter = logging.Formatter(cfg_logging["format"])

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    root_logger.setLevel(min(file_level, console_level))

    if out_dir is not None:
        _logging_file = os.path.join(
            out_dir, cfg_logging.get("filename", "logging.log")
        )
        file_handler = logging.FileHandler(_logging_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_level)
        root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Avoid pollution by packages
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)


class MyTrainingLogger:
    """Tensorboard + wandb logger"""

    writer: SummaryWriter
    is_initialized = False

    def __init__(self) -> None:
        pass

    def set_dir(self, tb_log_dir):
        if self.is_initialized:
            raise ValueError("Do not initialize writer twice")
        self.writer = SummaryWriter(tb_log_dir)
        self.is_initialized = True

    def log_dic(self, scalar_dic, global_step, walltime=None):
        for k, v in scalar_dic.items():
            self.writer.add_scalar(k, v, global_step=global_step, walltime=walltime)
        return


# global instance
tb_logger = MyTrainingLogger()


# -------------- wandb tools --------------
def init_wandb(enable: bool, **kwargs):
    if enable:
        run = wandb.init(sync_tensorboard=True, **kwargs)
    else:
        run = wandb.init(mode="disabled")
    return run


def log_slurm_job_id(step):
    global tb_logger
    _jobid = os.getenv("SLURM_JOB_ID")
    if _jobid is None:
        _jobid = -1
    tb_logger.writer.add_scalar("job_id", int(_jobid), global_step=step)
    logging.debug(f"Slurm job_id: {_jobid}")


def load_wandb_job_id(out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "r") as f:
        wandb_id = f.read()
    return wandb_id


def save_wandb_job_id(run, out_dir):
    with open(os.path.join(out_dir, "WANDB_ID"), "w+") as f:
        f.write(run.id)


def eval_dic_to_text(val_metrics: dict, dataset_name: str, sample_list_path: str):
    eval_text = f"Evaluation metrics:\n\
     on dataset: {dataset_name}\n\
     over samples in: {sample_list_path}\n"

    eval_text += tabulate([val_metrics.keys(), val_metrics.values()])
    return eval_text
