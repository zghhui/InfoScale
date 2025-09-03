accelerate launch --config_file ./multi_gpu.yaml --main_process_port 1233 text2image_xl.py \
    --pretrained_model_name_or_path ${your-stabilityai--stable-diffusion-2-1-base} \
    --validation_prompt ${your-validation_prompt} \
    --seed 123 \
    --config ./configs/sdxl_2048x2048.yaml \
    --logging_dir ./result/sdxl_2048x2048 \
    --gpu_num 4