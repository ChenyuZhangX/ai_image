export CUDA_VISIBLE_DEVICES=0
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop4" --to_drop 4