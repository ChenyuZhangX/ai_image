export CUDA_VISIBLE_DEVICES=2
python src/train_ablation.py --batch_size 350 --config "./config/config_transformer.json" --project "ablation1/drop5"