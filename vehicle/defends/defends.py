from typing import Any, Dict
import torch
import torch.nn as nn
import torch.optim as optim

import os
import yaml
from easydict import EasyDict

from ultralytics import YOLO

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_yolo_dataset, ClassificationDataset, build_dataloader
from ultralytics.utils import TQDM, emojis
from ultralytics.utils.torch_utils import select_device

# from ultralytics.models.classify. import ClassificationValidator

from nudt_ultralytics.callbacks.callbacks import callbacks_dict

from utils.sse import sse_clean_samples_gen_validated

# from defends.ipeg_compression import JpegCompression
from defends.jpeg_scale import JpegScale

class defends:
    def __init__(self, cfg, args):
        self.cfg = cfg
        # self.model = YOLO(model=cfg.model, task=cfg.task, verbose=cfg.verbose).load(cfg.pretrained)  # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        # self.model = YOLO(model=cfg.pretrained, task=cfg.task, verbose=cfg.verbose) # task: 'detect', 'segment', 'classify', 'pose', 'obb'. verbose: Display model info on load.
        # for (event, func) in callbacks_dict.items():
        #     self.model.add_callback(event, func)
        
        # self.model.overrides = cfg
        self.device = cfg.device
        
        if args.defend_method == 'comp':
            self.defend = JpegCompression(
                                clip_values=(0, 255),
                                quality=50,
                                channels_first=False,
                                apply_fit=True,
                                apply_predict=True,
                                verbose=False,
                            )
        elif args.defend_method == 'scale':
            self.defend = JpegScale(
                                scale=0.9,
                                interp="bilinear"
                            )
        else:
            raise ValueError('Invalid attach method!')
            
        
            
    def get_dataloader(self) -> torch.utils.data.DataLoader:
        if str(self.cfg.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
            data = check_det_dataset(self.cfg.data)
            dataset = build_yolo_dataset(self.cfg, data.get(self.cfg.split), self.cfg.batch, data, mode="val", stride=self.model.stride)
            dataloader = build_dataloader(dataset, self.cfg.batch, self.cfg.workers, shuffle=False, rank=-1, drop_last=self.cfg.compile, pin_memory=False)
        elif self.cfg.task == "classify":
            data = check_cls_dataset(self.cfg.data, split=self.cfg.split)
            dataset = ClassificationDataset(root=data.get(self.cfg.split), args=self.cfg, augment=self.cfg.augment, prefix=self.cfg.split)
            dataloader = build_dataloader(dataset, self.cfg.batch, self.cfg.workers, rank=-1)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.cfg.data}' for task={self.cfg.task} not found ❌"))

        return dataloader
    
    def get_desc(self) -> str:
        """Return a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        # batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        # batch["img"] = batch["img"].half() if self.cfg.half else batch["img"].float()
        # batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def run_defend(self, args):
        os.makedirs(f'{self.cfg.save_dir}/clean_images', exist_ok=True)
        dataloader = self.get_dataloader()
        desc = self.get_desc()
        bar = TQDM(dataloader, desc=desc, total=len(dataloader))
        for batch_i, batch in enumerate(bar):
            batch = self.preprocess(batch)
            if args.defend_method == 'comp':
                clean_image, _ = self.defend(batch["img"].numpy())
            elif args.defend_method == 'scale':
                clean_image, _ = self.defend(batch["img"])
            else:
                raise ValueError('Invalid attach method!')
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(clean_image[0])
            clean_image_name = f'{self.cfg.save_dir}/clean_images/clean_image_{batch_i}.jpg'
            pil_image.save(clean_image_name)
            sse_clean_samples_gen_validated(clean_image_name)
        