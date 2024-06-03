import torch
import pickle
from torch.utils.data import DataLoader

import os

from model.model_wraper import ModelWrapper
from model.encoder import Encoder_CNN
from model.decoder import Decoder_FC
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
    args = parser.parse_args()
    return args

def test(cfg):
    test_loader = DataLoader(DataSet_Scene(cfg['train_cfg']['data_path'], device = 'cpu', mode = 'test'), batch_size=100, shuffle=False)
    encoder = Encoder_CNN(cfg['encoder_cfg'])
    decoder = Decoder_FC(cfg['decoder_cfg'])
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
    probs = probs.cpu().numpy() # bs, 6
    preds = preds.cpu().numpy().reshape(-1) # bs
    gts = gts.cpu().numpy().reshape(-1)     # bs

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
    import matplotlib.pyplot as plt

    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average='macro')
    precision = precision_score(gts, preds, average='macro')
    recall = recall_score(gts, preds, average='macro')

    save_path = os.path.join(cfg['train_cfg']['save_path'], "metrics")
    os.makedirs(save_path, exist_ok=True)

    # confusion matrix
    num_classes = 6
    cm = confusion_matrix(gts, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(range(num_classes), range(num_classes))
    plt.yticks(range(num_classes), range(num_classes))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))

    # save metrics
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")



    print("Testing complete!")

if __name__ == "__main__":
    args = arg_parser()
    with open(args.config, 'rb') as f:
        cfg = pickle.load(f)

    cfg['test_cfg'] = {
        'ckpt': args.ckpt
    }
    test(cfg)




