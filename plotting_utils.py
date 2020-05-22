import argparse, os
import numpy as np
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
plt.style.use('ggplot')

def save_loss_curve(ckpt_path):
    img_path = ckpt_path[:ckpt_path.rfind('.')]+".png"
    if os.path.exists(img_path):
        input(f"{img_path} already exists. Continue? >> ")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    assert 'loss_curve' in ckpt
    
    losses = ckpt['loss_curve']
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    plt.title(f'Loss curve plot', fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average loss", fontsize=12)

    plt.plot(np.arange(len(losses)), losses, linestyle='-')
    # plt.legend(loc='upper left',
    #            borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='') 
    args = parser.parse_args()

    save_loss_curve(args.ckpt_path)