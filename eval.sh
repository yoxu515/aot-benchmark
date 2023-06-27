exp="ms_r50_v"
config=$exp
gpu_num="4"
dataset="viposeg"
split="val"
ckpt_path="pretrain_models/R50_AOTv3_PRE_VIP.pth"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path

exp="ms_swinb_v"
config=$exp
gpu_num="4"
dataset="viposeg"
split="val"
ckpt_path="pretrain_models/SwinB_AOTv3_PRE_VIP.pth"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path

exp="pano_r50"
config=$exp
gpu_num="4"
dataset="viposeg"
split="val"
ckpt_path="pretrain_models/R50_PAOT_PRE_VIP.pth"
python tools/eval.py --ema --exp_name ${exp} --config ${config}\
	--dataset ${dataset} --split ${split}  --gpu_num ${gpu_num} --ckpt_path $ckpt_path