export CUDA_VISIBLE_DEVICES=0
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop0" --to_drop 0
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop1" --to_drop 1
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop2" --to_drop 2
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop3" --to_drop 3
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop4" --to_drop 4
python src/train_ablation.py --batch_size 350 --config "./config/config_ablation1.json" --project "ablation1/drop5" --to_drop 5