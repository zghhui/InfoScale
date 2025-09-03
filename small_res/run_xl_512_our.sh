# 2048x2048 (4x) generation
python3 text2image_xl_our.py \
    --pretrained_model_name_or_path /home/mlsnrs/data/zyz/zgh/AIGC/FreeTailor/pretrain_model/models--stabilityai--stable-diffusion-xl-base-1.0 \
    --validation_prompt "/home/mlsnrs/data/zyz/zgh/AIGC/High_resulation/paper_experiments/method_compare/sdxl_small/prompt.txt" \
    --seed 23 \
    --config ./configs/sdxl_512x512.yaml \
    --logging_dir ./result/sdxl_512x512_our

        # --config ./configs/sdxl_512x512.yaml \