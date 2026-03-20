DIMENSION='subject_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality'
VIDEOS_PATH='/workspace/VBench/test_videos'
PROMPT_FILE=$VIDEOS_PATH/captions_vbench.json

python evaluate.py \
    --dimension $DIMENSION \
    --videos_path $VIDEOS_PATH \
    --prompt_file $PROMPT_FILE \
    --mode=custom_input