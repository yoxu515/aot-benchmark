## Inference ##

exp = "deafult"
data_path = "./datasets/Demo"
model = "r50_aotl"
ckpt_path = "./pretrained_models/R50_AOTL_PRE_YTB_DAV.pth"

python tools/demo.py --amp \
    --stage pre_ytb_dav \
    --model ${model} \
    --data_path ${data_path} \
    --ckpt_path ${ckpt_path} \

