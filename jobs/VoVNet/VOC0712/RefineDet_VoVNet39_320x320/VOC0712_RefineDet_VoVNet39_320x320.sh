cd /home/ywlee/VoVNet-RefineDet
./build/tools/caffe train \
--solver="models/VoVNet/VOC0712/RefineDet_VoVNet39_320x320/solver.prototxt" \
--weights="models/ImageNet/VoVNet/VoVNet39_ImageNet_Pretrained.caffemodel" \
--gpu 0,1,2,3 2>&1 | tee jobs/VoVNet/VOC0712/RefineDet_VoVNet39_320x320/VOC0712_RefineDet_VoVNet39_320x320.log
