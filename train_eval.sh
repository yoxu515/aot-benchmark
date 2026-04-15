exp="default"
gpu_num="4"
#in case of no gpu, just don't include this argument

model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_aotl"
# model="swinb_aotl"

## Training ##
stage="pre"
python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num}

stage="pre_ytb_dav"
# Single GPU
python tools/train.py --exp_name ${exp} --stage ${stage} --model ${model}

# Multi-GPU
torchrun --nproc_per_node=4 tools/train.py --exp_name ${exp} --stage ${stage} --model ${model}


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