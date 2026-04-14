exp="default"
gpu_num="4"
#in case of no gpu, just don't include this argument

model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_aotl"
# model="swinb_aotl"

## Pre-Training ##
stage="pre"
python tools/pretrain.py --amp \
	--exp_name ${exp} \
	--stage {stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \

## Training ##
pretrain_ckpt="pretrain_model/static_pretrain_final.pth"
stage="pre_ytb_dav"
python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \
	--pretrained_path ${pretrain_ckpt}

## Evaluation ##
dataset="davis2017"
split="test"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}

dataset="davis2017"
split="val"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}

dataset="davis2016"
split="val"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}

dataset="youtubevos2018"
split="val"  # or "val_all_frames"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}

dataset="youtubevos2019"
split="val"  # or "val_all_frames"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}