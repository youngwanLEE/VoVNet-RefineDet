# RefineDet with [VoVNet](https://arxiv.org/abs/1904.09730)(CVPRW'19) Backbone Networks for Real-time Object Detection

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This repository contains [RefineDet](https://github.com/sfzhang15/RefineDet) with [VoVNet](https://arxiv.org/abs/1904.09730) Backbone Networks in the following paper
[An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection](https://arxiv.org/abs/1904.09730) ([CVPRW'19](http://www.ee.oulu.fi/~lili/CEFRLatCVPR2019.html), oral)

## Highlights

- Memory & Energy efficient 
- Better performance, especially for *small* objects
- Faster speed


<p align="center">
<img src="https://github.com/youngwanLEE/VoVNet-RefineDet/blob/master/OSA.PNG" alt="OSA" width="500px">
</p>

<p align="center">
<img src="https://github.com/youngwanLEE/VoVNet-RefineDet/blob/master/coco_results.PNG" alt="coco_results" width="800px">
</p>

<p align="center">
<img src="https://github.com/youngwanLEE/VoVNet-RefineDet/blob/master/VOC_results.PNG" alt="VOC_results" width="800px">
</p>


### Installation


1. Get the code. We will call the cloned directory as `$RefineDet_ROOT`.
  ```Shell
  git clone https://github.com/youngwanLEE/VoVNet-RefineDet.git
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  cd $RefineDet_ROOT
  # Modify Makefile.config according to your Caffe installation.
  # Make sure to include $RefineDet_ROOT/python to your PYTHONPATH.
  cp Makefile.config.example Makefile.config
  make all -j && make py
  ```

### Preparation
1. Download [VoVNet39-ImageNet](https://dl.dropbox.com/s/zbys2uzvpfi7ko4/VoVNet39_ImageNet_Pretrained.caffemodel?dl=1). By default, we assume the model is stored in `$RefineDet_ROOT/models/ImageNet/VoVNet/`.

3. Follow the [data/VOC0712/README.md](https://github.com/sfzhang15/RefineDet/blob/master/data/VOC0712/README.md) to download VOC2007 and VOC2012 dataset and create the LMDB file for the VOC2007 training and testing.

4. Follow the [data/VOC0712Plus/README.md](https://github.com/sfzhang15/RefineDet/blob/master/data/VOC0712Plus/README.md) to download VOC2007 and VOC2012 dataset and create the LMDB file for the VOC2012 training and testing.

5. Follow the [data/coco/README.md](https://github.com/sfzhang15/RefineDet/blob/master/data/coco/README.md) to download MS COCO dataset and create the LMDB file for the COCO training and testing.

### Training
1. Train your model on PASCAL VOC.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $RefineDet_ROOT/models/VoVNet/VOC0712{Plus}/refinedet_vovnet39_{size}x{size}/
  # and job file, log file, and the python script in:
  #   - $RefineDet_ROOT/jobs/VoVNet/VOC0712{Plus}/refinedet_vovnet39_{size}x{size}/
  python examples/refinedet/VoVNet39_VOC2007_320.py
  python examples/refinedet/VoVNet39_VOC2007_512.py
  ```
  
2. Train your model on COCO.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $RefineDet_ROOT/models/VoVNet/coco/refinedet_vovnet39_{size}x{size}/
  # and job file, log file, and the python script in:
  #   - $RefineDet_ROOT/jobs/VoVNet/coco/refinedet_vovnet39_{size}x{size}/
  python examples/refinedet/VoVNet39_COCO_320.py
  python examples/refinedet/VoVNet39_COCO_512.py
  ```




### Evaluation
1. Build the Cython modules.
  ```Shell
  cd $RefineDet_ROOT/test/lib
  make -j
  ```
  
2. Change the ‘self._devkit_path’ in [`test/lib/datasets/pascal_voc.py`](https://github.com/sfzhang15/RefineDet/blob/master/test/lib/datasets/pascal_voc.py) to yours.

3. Change the ‘self._data_path’ in [`test/lib/datasets/coco.py`](https://github.com/sfzhang15/RefineDet/blob/master/test/lib/datasets/coco.py) to yours.

4. Check out [`test/refinedet_demo.py`](https://github.com/sfzhang15/RefineDet/blob/master/test/refinedet_demo.py) on how to detect objects using the RefineDet model and how to plot detection results.
  ```Shell
  # For GPU users
  python test/refinedet_demo.py
  # For CPU users
  python test/refinedet_demo.py --gpu_id -1
  ```

5. Evaluate the trained models via [`test/refinedet_test.py`](https://github.com/sfzhang15/RefineDet/blob/master/test/refinedet_test.py).
  ```Shell
  # You can modify the parameters in refinedet_test.py for different types of evaluation:
  #  - single_scale: True is single scale testing, False is multi_scale_testing.
  #  - test_set: 'voc_2007_test', 'voc_2012_test', 'coco_2014_minival', 'coco_2015_test-dev'.
  #  - voc_path: where the trained voc caffemodel.
  #  - coco_path: where the trained voc caffemodel.
  # For 'voc_2007_test' and 'coco_2014_minival', it will directly output the mAP results.
  # For 'voc_2012_test' and 'coco_2015_test-dev', it will save the detections and you should submitted it to the evaluation server to get the mAP results.
  python test/refinedet_test.py
  ```

### Models

1. PASCAL VOC models :
   * 07+12: [VoVNet39-RefineDet320](https://www.dropbox.com/sh/epllpxx3pxfl9yd/AAAaHXhsAxeDDctmMou86LoWa?dl=1),   [VoVNet39-RefineDet512](https://www.dropbox.com/sh/wla1bijjr8ql2m3/AADx_UFyewocAGupg_U2ZjOva?dl=1)


2. COCO models:
   * trainval35k: [VoVNet39-RefineDet320](https://www.dropbox.com/sh/sjg5lagetlada9w/AAAXUZ24w0P5TuY210VkwPH9a?dl=1), [VoVNet39-RefineDet512](https://www.dropbox.com/sh/1kt6w1cfxxjpp02/AABjGrfb_jY50iYddiAlfQBUa?dl=1)



### Citing VoVNet

Please cite our paper in your publications if it helps your research:

    @inproceedings{lee2019energy,
      title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
      author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
      year = {2019}
    }