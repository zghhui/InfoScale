# 2048x2048 (4x) generation
python3 text2image.py \
    --pretrained_model_name_or_path /home/mlsnrs/data/zyz/zgh/AIGC/FreeTailor/pretrain_model/models--stabilityai--stable-diffusion-2-1-base \
    --validation_prompt "/home/mlsnrs/data/zyz/zgh/AIGC/High_resulation/laion_test/merged.txt" \
    --seed 123 \
    --config ./configs/sd2.1_1024x1024.yaml \
    --logging_dir ./high_logging_dir_1024x1024