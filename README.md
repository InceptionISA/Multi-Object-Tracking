# Fine-tunied bytetrack_x_mot17 on Mot20

## Installation

### 1. Installing on the host machine

Step1. Install ByteTrack.

```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others

```shell
pip3 install cython_bbox
```

<!-- ### 2. Docker build

```shell
docker build -t bytetrack:latest .

# Startup sample
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/ByteTrack/pretrained \
-v $PWD/datasets:/workspace/ByteTrack/datasets \
-v $PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
bytetrack:latest
``` -->

## Model preparation

download the model bytetrack_x_mot17 :

```shell
gdown --id 1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5 -O pretrained/
```

## Data preparation

download the data from these link [MOT20](https://motchallenge.net/) :

```shell
wget https://motchallenge.net/data/MOT20.zip -O MOT20.zip


unzip MOT20.zip -d datasets
rm -rf  datasets/MOT20/train/MOT20-01
rm MOT20.zip

```

```
https://motchallenge.net/data/MOT20.zip

then remove the mot20-01

datasets
|
└——————MOT20
        └——————train
        └——————test

```

Then, you need to turn the dataset to COCO format :

```shell
python3 tools/convert_mot20_to_coco.py

```

## Training

- **Train MOT20 test model (MOT20 train, CrowdHuman)**

For MOT20, you need to clip the bounding boxes inside the image.

Add clip operation in [line 134-135 in data_augment.py](https://github.com/ifzhang/ByteTrackblob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/data_augment.py#L134), [line 122-125 in mosaicdetection.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L122), [line 217-225 in mosaicdetection.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/data/datasets/mosaicdetection.py#L217), [line 115-118 in boxes.py](https://github.com/ifzhang/ByteTrack/blob/72cd6dd24083c337a9177e484b12bb2b5b3069a6/yolox/utils/boxes.py#L115).

```shell
python tools/train.py -f exps/example/mot/yolox_x_mix_mot20.py  -b 4 --fp16 -o -c pretrained/bytetrack_x_mot17.pth.tar
```

<!--
- **Train custom dataset**

First, you need to prepare your dataset in COCO format. You can refer to [MOT-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_mot17_to_coco.py) or [CrowdHuman-to-COCO](https://github.com/ifzhang/ByteTrack/blob/main/tools/convert_crowdhuman_to_coco.py). Then, you need to create a Exp file for your dataset. You can refer to the [CrowdHuman](https://github.com/ifzhang/ByteTrack/blob/main/exps/example/mot/yolox_x_ch.py) training Exp file. Don't forget to modify get_data_loader() and get_eval_loader in your Exp file. Finally, you can train bytetrack on your dataset by running:

```shell
cd <ByteTrack_HOME>
python3 tools/train.py -f exps/example/mot/your_exp_file.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
``` -->
