from FaceDetection.darkflow.net.build import TFNet
import cv2
import numpy as np

import tensorflow as tf

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

options = {
    "model": "cfg/tiny-yolo-voc-new.cfg",
    "load": 14050,
    "lr": 0.000001,
    "batch": 8,
    "epoch": 100,
    "gpu": 0.8,
    "backup": "./ckpt/",
    "save": 200,
    "train": True,
    "annotation": "./dataset_extended/annotations",
    "dataset": "./dataset_extended/faces"
}

tfnet = TFNet(options)
tfnet.train()
