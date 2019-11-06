./build/examples/demo/video_detect.bin \
models/VoVNet/coco/refinedet_VoVNet27_320x320/merge_bn_deploy.prototxt \
models/VoVNet/coco/refinedet_VoVNet27_320x320/merge_bn_weight.caffemodel \
examples/demo/list_demo_files.txt \
examples/demo/results/RefineDet_D3FixG_51x512_TT001.wmv \
0

# ./build/examples/demo/video_detect.bin \
# models/DiverseNet/coco/refinedet_hyper_v3_D3_FixG_Conv-NoBN-ReLU_512x512/deploy.prototxt \
# models/DiverseNet/coco/refinedet_hyper_v3_D3_FixG_Conv-NoBN-ReLU_512x512/coco_refinedet_hyper_v3_D3_FixG_Conv-NoBN-ReLU_512x512_final.caffemodel \
# examples/refinedet/list_demo_files.txt \
# examples/demo/results/RefineDet_D3FixG_51x512_TT001.wmv \
# 0


# ./build/examples/demo/video_detect.bin \
# models/ResNet/coco/refinedet_resnet101_512x512/deploy.prototxt \
# models/ResNet/coco/refinedet_resnet101_512x512/coco_refinedet_resnet101_512x512_final.caffemodel \
# examples/refinedet/list_demo_files.txt \
# examples/demo/results/RefineDet_ResNet101_512x512_TT001.wmv \
# 1

#For SSD
# ./build/examples/demo/video_detect.bin \
# /home/ywlee/caffe/models/WR-Inception/coco/DSSD_WRInception_v2_19_IREv4_PM_HDM_freezedBN_smaller_batch20_512x512/deploy.prototxt \
# /home/ywlee/caffe/models/WR-Inception/coco/DSSD_WRInception_v2_19_IREv4_PM_HDM_freezedBN_smaller_batch20_512x512/WRInception_coco_DSSD_WRInception_v2_19_IREv4_PM_HDM_freezedBN_smaller_batch20_512x512_iter_400000.caffemodel \
# examples/refinedet/list_demo_files.txt \
# examples/demo/results/SSD_WRI_TT001.wmv


# ./build/examples/demo/video_detect.bin \
# /home/ywlee/caffe/models/DiverseNet/coco/SSD_Hyper_v3_D3_FixG_Conv-NoBN-ReLU_ExtraResOrigin_512x512/deploy.prototxt \
# /home/ywlee/caffe/models/DiverseNet/coco/SSD_Hyper_v3_D3_FixG_Conv-NoBN-ReLU_ExtraResOrigin_512x512/_coco_SSD_Hyper_v3_D3_FixG_Conv-NoBN-ReLU_ExtraResOrigin_512x512_iter_480000.caffemodel \
# examples/refinedet/list_demo_files.txt \
# examples/demo/results/SSD_VoVNet_TT001.wmv