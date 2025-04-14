# encoding: utf-8
import os
import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1  # MOT20 has only one class (person)
        self.depth = 1.33     # Depth multiplier for YOLOX-X
        self.width = 1.25     # Width multiplier for YOLOX-X
        self.random_size = (20, 28)  # Keeps images smaller

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"  # COCO-format annotations for MOT20 train
        self.val_ann = "train.json"     # COCO-format annotations for MOT20 validation
        self.input_size = (448, 800)  # Input size for training
        self.test_size = (448, 800)   # Input size for testing
        self.max_epoch = 20            # Fewer epochs for fine-tuning
        self.print_interval = 200        # Print loss every 20 iterations
        self.eval_interval = 99         # Evaluate every 99 epochs
        self.test_conf = 0.05          # Confidence threshold for testing
        self.nmsthre = 0.6             # NMS threshold
        self.no_aug_epochs = 7         # Disable data augmentation in the last epochs
        self.basic_lr_per_img = 0.0001
        self.warmup_epochs = 3         # Warmup epochs


    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "MOT20"),
            json_file=self.train_ann,
            name='train',  # Use 'train' for training
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=600,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1200,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "MOT20"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train', # change to train when running on training set	
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator