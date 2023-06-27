# Pretraining stage
exp="ms_r50_yd"
pretrain="pretrain_models/resnet50-0676ba61.pth"
stage="PRE"
python tools/train.py --amp \
	--exp_name ${exp} \
	--config ${exp} \
	--stage ${stage} \
  	--pretrained_path ${pretrain}

# Main training stage
exp="ms_r50_yd"
pretrain="pretrain_models/R50_AOTv3_PRE.pth"
stage="PRE_YTB_DAV"
python tools/train.py --amp \
	--exp_name ${exp} \
	--config ${exp} \
	--stage ${stage} \
	--pretrained_path ${pretrain}