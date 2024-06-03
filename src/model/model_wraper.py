
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import Optional
from torch import nn, optim
import tqdm

import pandas as pd
import matplotlib.pyplot as plt

class ModelWrapper:
    logger: Optional[WandbLogger]
    encoder: nn.Module
    decoder: nn.Module
    losses: nn.ModuleList
    # optimizer_cfg: OptimizerCfg
    # test_cfg: TestCfg
    # train_cfg: TrainCfg
    # step_tracker: StepTracker | None

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        losses: list[nn.Module],
        optimizer: optim.Optimizer,
        cfg, # config will be added later
        logger: Optional[WandbLogger],
        device: str,
    ) -> None:
        super().__init__()
        self.device = device

        # cfg 
        self.cfg = cfg
        
        # Set up the model.
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.losses = nn.ModuleList(losses).to(device)

        # hyperparams
        self.use_pred_bw = self.cfg['train_cfg'].get('use_pred_bw', False)
        self.num_classes = self.cfg['decoder_cfg'].get('num_classes', 6)

        # optimizer
        if optimizer:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())

            self.optimizer = optimizer(
                params,
                lr=self.cfg['optimizer_cfg']['lr'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=1e-3,
            )
        # Set up the logger.
        self.logger = logger

    def training_step(self, 
                      dataloader: DataLoader,
                      eval_loader: DataLoader = None,
                      ):
        
        save_path = self.cfg['train_cfg']['save_path']
        os.makedirs(save_path, exist_ok=True)

        # new txt
        with open(os.path.join(save_path, 'evals.txt'), 'w') as f:
            f.write('Eval Metrics\n')
        with open(os.path.join(save_path, 'test.txt'), 'w') as f:
            f.write('Test Metrics\n')

        self.train()
        
        loss2report = []
        acc2report = []
        for epoch in tqdm.tqdm(range(self.cfg['train_cfg']['num_epochs'])):
            running_loss = 0.0
            pos = 0
            total = 0
            for _, data in enumerate(dataloader):
                x = data['image']
                label_gt = data['label']

                # to device
                x = x.to(self.device)
                label_gt = label_gt.to(self.device)
                
                # Forward pass.
                latent = self.encoder(x)
                # decoder
                label_pred = self.decoder(latent)
                # loss part 
                lams = [lam for lam in self.cfg['loss_cfg'].values()]
                
                # Compute the loss.
                loss = 0
                for loss_fn, lam in zip(self.losses, lams):
                    loss += loss_fn(label_pred, F.one_hot(label_gt, num_classes = 6).float()) * lam

                label_pred = torch.argmax(label_pred, dim=1)
                pos += torch.where(label_pred == label_gt)[0].shape[0]
                total += label_gt.shape[0]

                # Backward pass.
                loss.backward()

                # Update the weights.
                self.optimizer.step()

                # Zero the gradients.
                self.optimizer.zero_grad()

                # Log the loss.
                running_loss += loss.item()

            acc = pos / total
            acc2report.append(acc)
            loss2report.append(running_loss)
            
            if self.logger is not None:
                self.logger.log_metrics({'loss': running_loss}, step=epoch)
            else:
                tqdm.tqdm.write(f'Epoch: {epoch} Loss: {running_loss} Acc: {acc}')
            

            if epoch % 100 == 0:
                if epoch % 1000 == 0:
                    os.makedirs(os.path.join(save_path, f'step{epoch}~{epoch+1000}'), exist_ok=True)
                info_path = os.path.join(save_path, f'step{epoch//1000 * 1000}~{epoch//1000 * 1000+1000}')
                # eval
                if eval_loader is not None:
                    self.eval_step(eval_loader, epoch = epoch)
                    self.train()
                self.save(os.path.join(info_path, 'model.pth'))
                self.plot(torch.tensor(loss2report), torch.tensor(acc2report), os.path.join(info_path, 'loss_acc.png'))
        
        
        
    
    def eval_step(self, eval_loader, **kwargs):

        self.eval()
        save_path = os.path.join(self.cfg['train_cfg']['save_path'], "evals.txt")

        pos = 0
        total = 0
        for _, data in enumerate(eval_loader):

            x = data['image']
            label_gt = data['label']

            x = x.to(self.device)
            label_gt = label_gt.to(self.device)
            
            # Forward pass.
            latent = self.encoder(x)
            # decoder
            label_pred = self.decoder(latent)

            # ACC
            label_pred = torch.argmax(label_pred, dim=1)
            pos += torch.where(label_pred == label_gt)[0].shape[0]
            total += label_gt.shape[0]
        
        acc = pos / total
        print(f"Eval acc: {acc}")
        with open(save_path, 'a') as f:
            f.write(f"Epoch: {kwargs.get('epoch', -1)} Eval acc: {acc}\n")


    def test_step(self, test_loader):
        
        self.eval()

        probs = torch.empty(0, self.num_classes).cuda()
        preds = torch.empty(0, 1).cuda()
        labels_gt = torch.empty(0, 1).cuda()

        for _, data in enumerate(test_loader):

            x = data['image']
            label_gt = data['label']

            x = x.to(self.device)
            
            label_gt = label_gt.to(self.device)
            labels_gt = torch.cat((labels_gt, label_gt.unsqueeze(1)), dim=0)
            
            # Forward pass.
            latent = self.encoder(x)
            # decoder
            pred = self.decoder(latent)
            probs = torch.cat((probs, pred), dim=0)

            # ACC
            label_pred = torch.argmax(pred, dim=1).unsqueeze(1)
            preds = torch.cat((preds, label_pred), dim=0)
               
        return preds, probs, labels_gt

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def save(self, path: str):
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
        }

        torch.save(
            checkpoint,
            path,
        )
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def plot(self, loss2report, acc2report, path):
        plt.plot(loss2report / torch.max(loss2report), label = 'loss')
        plt.plot(acc2report, label = 'acc')
        # add text
        plt.xlabel('step')
        plt.ylabel('loss/acc')
        # max acc
        plt.text(torch.argmax(acc2report), torch.max(acc2report), f"max acc: {torch.max(acc2report)}")

        plt.legend()
        plt.savefig(path)
        plt.close()


