import torch
import pickle
from torch.utils.data import DataLoader

import os

from model.model_wraper import ModelWrapper
from model.encoder import Encoder_CNN
from model.decoder import Decoder_Transformer
from utils import get_metrics
from loss import LossCLS
from dataset.dataset_scene import DataSet_Scene
import argparse
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def arg_parser():
    parser = argparse.ArgumentParser(description='Training and Testing the model')
    parser.add_argument('--config', type=str, default='./outputs/exp1/config.pkl', help='Path to the config file')
    parser.add_argument('--ckpt', type=str, default='./outputs/exp1/ckpt.pth', help='Path to the checkpoint file')
    parser.add_argument('--batch_size', type=int, default=350, help='Batch size for training')
    parser.add_argument('--to_drop', type=int, default=5, help='Label to drop')
    args = parser.parse_args()
    return args

def test(cfg):
    test_loader = DataLoader(DataSet_Scene(cfg['train_cfg']['data_path'], device = 'cpu', mode = 'test', ablation1 = True, to_drop = cfg['to_drop']), batch_size=100, shuffle=False)
    encoder = Encoder_CNN(cfg['encoder_cfg'])
    decoder = Decoder_Transformer(cfg['decoder_cfg'])
    losses = [
              LossCLS(),
                ]
    
    ckpt = cfg['test_cfg']['ckpt']

    model = ModelWrapper(encoder, decoder, losses, None, cfg, logger=None, device=device)
    model.load(ckpt)

    # no grad
    with torch.no_grad():
        preds, probs, gts = model.test_step(test_loader)

    # as numpy 
    probs = probs.cpu().numpy() # bs, num_classes
    preds = preds.cpu().numpy().reshape(-1) # bs
    gts = gts.cpu().numpy().reshape(-1)     # bs

    # calculate_metrics
    get_metrics(preds, probs, gts, os.path.join(cfg['train_cfg']['save_path'], "metrics"))

    print("Testing complete!")



    


if __name__ == "__main__":
    args = arg_parser()
    with open(args.config, 'rb') as f:
        cfg = pickle.load(f)

    cfg['test_cfg'] = {
        'ckpt': args.ckpt
    }
    cfg['to_drop'] = args.to_drop
    test(cfg)




