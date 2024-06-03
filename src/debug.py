
import os
import torch
from einops import rearrange
from torch import nn
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Debugging')
    # indexes mutilple integers, for short -i
    parser.add_argument('-i', '--idx', type=int, default=0, help='Index')
    args = parser.parse_args()
    return args

def debug():
    x = torch.randn((20, 60))
    print(x.shape)
    x = x.chunk(3, dim=-1)
    for idx in range(len(x)):
        print(x[idx].shape)
    print(torch.stack(x, dim=-1).shape)
    print(torch.cat(x, dim = -1).shape)

    qkv = x
    


def debug1():
    os.makedirs('debug', exist_ok=True)
    acc2report = []
    loss2report = []
    for epoch in range(101):
        loss2report.append(torch.rand(1).item())
        acc2report.append(torch.rand(1).item())

        if epoch % 10 == 0:
            if epoch % 100 == 0:
                os.makedirs(os.path.join('debug', f'step{epoch//100 * 100}~{epoch//100 * 100 +100}'), exist_ok=True)
            # plot loss and acc
            plt.plot(torch.tensor(loss2report) / torch.max(torch.tensor(loss2report)), label = 'loss')
            plt.plot(torch.tensor(acc2report), label = 'acc')
            # add text
            plt.xlabel('step')
            plt.ylabel('loss/acc')
            # max acc
            plt.text(torch.argmax(torch.tensor(acc2report)), torch.max(torch.tensor(acc2report)), f"max acc: {torch.max(torch.tensor(acc2report))}")
            plt.legend()
            plt.savefig(os.path.join('debug', f'step{epoch//100 * 100}~{epoch//100 * 100+100}', 'loss_acc.png'))
            plt.close()
            print(f'Epoch: {epoch} Loss: {loss2report[-1]} Acc: {acc2report[-1]}')

def debug2():
    # brownian motion
    os.makedirs('debug', exist_ok=True)
    x = [0]
    for t in range(101):
        x.append(x[-1] + torch.randn(1).item())
    plt.plot(torch.tensor(x))
    plt.savefig(os.path.join('debug', 'brownian_motion.png'))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    idx = args.idx
    if idx == 0:
        debug()
    else:
        eval(f'debug{idx}')()