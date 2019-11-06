# python tools/extra/pvanet_merge_bn.py '/home/ywlee/RefineDet/models/DiverseNet/VOC/refinedet_diverseNetV3_D4B_FixG64_Batch20_180k_@Min3_512x512/deploy.prototxt' '/home/ywlee/RefineDet/models/DiverseNet/VOC0712/refinedet_diverseNetV3_D4B_FixG64_Batch20_180k_@Min3_512x512/VOC0712_refinedet_diverseNetV3_D3_FixG64_Batch24_160k_@Min3_512x512_final.caffemodel' \
# --output_model '/home/ywlee/RefineDet/models/DiverseNet/coco/refinedet_hyper_v3_D4B_FixG64_20Batch_540k_512x512/PVANET_Merge_deploy.prototxt'   \
# --output_weights '/home/ywlee/RefineDet/models/DiverseNet/coco/refinedet_hyper_v3_D4B_FixG64_20Batch_540k_512x512/merge_bn_weight.caffemodel'

# python tools/extra/pvanet_merge_bn.py '/home/ywlee/RefineDet/models/DiverseNet/coco/refinedet_hyper_v3_D4B_FixG64_Conv-NoBN-ReLU_320x320' 
python tools/extra/merge_bn.py '/home/ywlee/RefineDet/models/VoVNet/coco/refinedet_VoVNet27_320x320'