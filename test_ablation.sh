export CUDA_VISIBLE_DEVICES=1
python src/test_ablation.py --config outputs/ablation1/drop0/config.pkl --ckpt outputs/ablation1/drop0/step5000~6000/model.pth --to_drop 0
python src/test_ablation.py --config outputs/ablation1/drop1/config.pkl --ckpt outputs/ablation1/drop1/step5000~6000/model.pth --to_drop 1
python src/test_ablation.py --config outputs/ablation1/drop2/config.pkl --ckpt outputs/ablation1/drop2/step5000~6000/model.pth --to_drop 2
python src/test_ablation.py --config outputs/ablation1/drop3/config.pkl --ckpt outputs/ablation1/drop3/step5000~6000/model.pth --to_drop 3
python src/test_ablation.py --config outputs/ablation1/drop4/config.pkl --ckpt outputs/ablation1/drop4/step5000~6000/model.pth --to_drop 4
python src/test_ablation.py --config outputs/ablation1/drop5/config.pkl --ckpt outputs/ablation1/drop5/step5000~6000/model.pth --to_drop 5