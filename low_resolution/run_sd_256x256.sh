# 256x256 (1/4x) generation
python3 text2image.py \
    --pretrained_model_name_or_path ${your-stabilityai--stable-diffusion-2-1-base} \
    --validation_prompt ${your-validation_prompt} \
    --seed 123 \
    --config ./configs/sd2.1_256x256.yaml \
    --logging_dir ${your-logging-dir}

    # --pretrained_model_name_or_path ${your-stabilityai--stable-diffusion-2-1-base} \
    # --validation_prompt ${your-validation_prompt} \