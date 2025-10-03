accelerate launch text2image_xl.py \
    --pretrained_model_name_or_path ${your-stabilityai--stable-diffusion-xl-base-1.0} \
    --validation_prompt ${your-validation_prompt} \
    --seed 123 \
    --config ./configs/sdxl_2048x2048.yaml \
    --logging_dir ./result/sdxl_2048x2048