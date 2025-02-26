import os
from helpers.config import base_dir , output_dir
import shutil


class YOLOPreprocessor:
    def __init__(self, base_dir, output_dir):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.train_img_dir = os.path.join(output_dir, "train/images")
        self.train_label_dir = os.path.join(output_dir, "train/labels")
        self.val_img_dir = os.path.join(output_dir, "val/images")
        self.val_label_dir = os.path.join(output_dir, "val/labels")
        self.test_img_dir = os.path.join(output_dir, "test/images")
        
        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(self.train_img_dir, exist_ok=True)
        os.makedirs(self.train_label_dir, exist_ok=True)
        os.makedirs(self.val_img_dir, exist_ok=True)
        os.makedirs(self.val_label_dir, exist_ok=True)
        os.makedirs(self.test_img_dir, exist_ok=True)
    
    def read_seqinfo(self, seqinfo_path):
        with open(seqinfo_path, "r") as f:
            lines = f.readlines()
        info = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in lines if '=' in line}
        return int(info["imWidth"]), int(info["imHeight"])

    def copy_images(self, src_folder, dest_folder, prefix=None):
        for root, _, files in os.walk(src_folder):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    new_name = f"{prefix}_{file}" if prefix else file
                    shutil.copy(os.path.join(root, file), os.path.join(dest_folder, new_name))
    
    def convert_gt_to_yolo(self, gt_path, label_folder, img_width, img_height, prefix):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]
        
        for img_id, group in df.groupby("frame"):
            img_name = f"{prefix}_{img_id:06d}.txt"  
            label_path = os.path.join(label_folder, img_name)
            with open(label_path, "w") as f:
                for _, row in group.iterrows():
                    cx = (row.x + row.w / 2) / img_width  
                    cy = (row.y + row.h / 2) / img_height  
                    w = row.w / img_width
                    h = row.h / img_height
                    f.write(f"{row['class']} {cx} {cy} {w} {h}\n")
    
    def process_data(self, train_seqs, val_seqs, test_seq):
        for seq in train_seqs:
            img_src = os.path.join(self.base_dir, "train", seq, "img1")
            gt_src = os.path.join(self.base_dir, "train", seq, "gt", "gt.txt")
            seqinfo_path = os.path.join(self.base_dir, "train", seq, "seqinfo.ini")
            img_width, img_height = self.read_seqinfo(seqinfo_path)
            self.copy_images(img_src, self.train_img_dir, seq)
            self.convert_gt_to_yolo(gt_src, self.train_label_dir, img_width, img_height, seq)
        
        for seq in val_seqs:
            img_src = os.path.join(self.base_dir, "train", seq, "img1")
            gt_src = os.path.join(self.base_dir, "train", seq, "gt", "gt.txt")
            seqinfo_path = os.path.join(self.base_dir, "train", seq, "seqinfo.ini")
            img_width, img_height = self.read_seqinfo(seqinfo_path)
            self.copy_images(img_src, self.val_img_dir, seq)
            self.convert_gt_to_yolo(gt_src, self.val_label_dir, img_width, img_height, seq)
        
        self.copy_images(os.path.join(self.base_dir, "test", test_seq, "img1"), self.test_img_dir)
        
        print("Dataset prepared for YOLO training and validation.")


yolo_preprocessor = YOLOPreprocessor(base_dir, output_dir)
yolo_preprocessor.process_data(train_seqs=["02", "05"], val_seqs=["03"], test_seq="01")
