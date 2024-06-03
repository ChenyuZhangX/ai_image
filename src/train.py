import torch
import pickle
from torch.utils.data import DataLoader

import os

from model.model_wraper import ModelWrapper
from model.encoder import Encoder_CNN
from model.decoder import Decoder_Transformer
from loss import LossCLS
from dataset.dataset_scene import DataSet_Scene
import argparse
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def arg_parser():
    parser = argparse.ArgumentParser(description='Training and Testing the model')
    parser.add_argument('--config', type=str, default='./config/config1.json', help='Path to the config file')
    parser.add_argument('--project', type=str, default='expx', help='Name of the project')
    parser.add_argument('--batch_size', type=int, default=350, help='Batch size for training')
    args = parser.parse_args()
    return args

def train(cfg):
    train_loader = DataLoader(DataSet_Scene(cfg['train_cfg']['data_path'], device = 'cpu', mode = 'train'), batch_size=cfg['train_cfg']['batch_size'], shuffle=True)
    val_loader = DataLoader(DataSet_Scene(cfg['train_cfg']['data_path'], device = 'cpu', mode = 'val'), batch_size=100, shuffle=False)
    test_loader = DataLoader(DataSet_Scene(cfg['train_cfg']['data_path'], device = 'cpu', mode = 'test'), batch_size=100, shuffle=False)

    encoder = Encoder_CNN(cfg['encoder_cfg'])
    decoder = Decoder_Transformer(cfg['decoder_cfg'])
    losses = [
              LossCLS(),
    ]
    optimizer = torch.optim.Adam

    path = cfg['train_cfg']['save_path']
    os.makedirs(path, exist_ok=True)
    # save cfg as pickle
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(cfg, f)

    model = ModelWrapper(encoder, decoder, losses, optimizer, cfg, logger=None, device=device)
    model.training_step(train_loader, eval_loader=val_loader)

    # no grad
    with torch.no_grad():
        model.test_step(test_loader)

if __name__ == "__main__":
    args = arg_parser()
    with open(args.config, 'r') as cfg:
        cfg = json.load(cfg)
    cfg['train_cfg']['batch_size'] = args.batch_size
    cfg['train_cfg']['save_path'] = os.path.join('./outputs', args.project)
    train(cfg)
    print("Training complete!")




