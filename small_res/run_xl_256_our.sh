# 2048x2048 (4x) generation
python3 text2image_xl_our.py \
    --pretrained_model_name_or_path /home/mlsnrs/data/zyz/zgh/AIGC/FreeTailor/pretrain_model/models--stabilityai--stable-diffusion-xl-base-1.0 \
    --validation_prompt "/home/mlsnrs/data/zyz/zgh/AIGC/High_resulation/paper_experiments/method_compare/sdxl_small/prompt_20.txt" \
    --seed 23 \
    --config ./configs/sdxl_256x256.yaml \
    --logging_dir ./result/sdxl_256x256_our

        # --config ./configs/sdxl_256x256.yaml \