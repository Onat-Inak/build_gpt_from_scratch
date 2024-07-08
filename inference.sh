# BUCKETDIR="gs://mbadas-prod-sandbox-research-bc0bb1f/workspace/project/build_gpt_from_scratch/models/build_gpt_bs64_n_256_iters_5000_eval_100/f480a8541a49e4beb847_2024_03_12_173432/model_weights.pth"
# MODELDIR="$HOME/data/inference_models/build_gpt_from_scratch" 

# gsutil -m cp -R  $BUCKETDIR $MODELDIR

python ./src/inference.py