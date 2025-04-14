import os
import numpy as np
import json
import cv2

# Define paths
DATA_PATH = "data/MOT20"
OUT_PATH = os.path.join(DATA_PATH, "annotations")

# Process only val_half
SPLIT = "val_half"

if __name__ == "__main__":
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    data_path = os.path.join(DATA_PATH, "train")  # Since val_half comes from train
    out_path = os.path.join(OUT_PATH, "{}.json".format(SPLIT))
    
    out = {
        "images": [],
        "annotations": [],
        "videos": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }

    seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1

    for seq in sorted(seqs):
        if ".DS_Store" in seq:
            continue

        video_cnt += 1
        out["videos"].append({"id": video_cnt, "file_name": seq})

        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, "img1")
        ann_path = os.path.join(seq_path, "gt/gt.txt")

        images = os.listdir(img_path)
        num_images = len([image for image in images if "jpg" in image]) 

        # Full sequence without splitting
        image_range = [0, num_images - 1]

        for i in range(num_images):
            img = cv2.imread(os.path.join(data_path, "{}/img1/{:06d}.jpg".format(seq, i + 1)))
            height, width = img.shape[:2]
            image_info = {
                "file_name": "{}/img1/{:06d}.jpg".format(seq, i + 1),
                "id": image_cnt + i + 1,
                "frame_id": i + 1,
                "prev_image_id": image_cnt + i if i > 0 else -1,
                "next_image_id": image_cnt + i + 2 if i < num_images - 1 else -1,
                "video_id": video_cnt,
                "height": height,
                "width": width,
            }
            out["images"].append(image_info)
        print("{}: {} images".format(seq, num_images))

        # Process annotations if the ground truth file exists
        if os.path.exists(ann_path):
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
            
            print("{} ann images".format(int(anns[:, 0].max())))
            for i in range(anns.shape[0]):
                frame_id = int(anns[i][0])

                track_id = int(anns[i][1])
                cat_id = int(anns[i][7])
                ann_cnt += 1

                if not (int(anns[i][6]) == 1):  # Ignore if not valid
                    continue
                if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person objects
                    continue
                if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored persons
                    continue

                category_id = 1  # pedestrian
                if not track_id == tid_last:
                    tid_curr += 1
                    tid_last = track_id

                ann = {
                    "id": ann_cnt,
                    "category_id": category_id,
                    "image_id": image_cnt + frame_id,
                    "track_id": tid_curr,
                    "bbox": anns[i][2:6].tolist(),
                    "conf": float(anns[i][6]),
                    "iscrowd": 0,
                    "area": float(anns[i][4] * anns[i][5]),
                }
                out["annotations"].append(ann)
        else:
            print(f"Warning: No annotation file found for {seq}, skipping annotations.")

        image_cnt += num_images
        print(tid_curr, tid_last)

    print("Loaded val_half for {} images and {} samples".format(len(out["images"]), len(out["annotations"])))
    json.dump(out, open(out_path, "w"))