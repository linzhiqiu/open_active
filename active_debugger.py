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
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close('all')

def save_acc_curve(ckpt_path):
    img_path = ckpt_path[:ckpt_path.rfind('.')]+"_acc.png"
    if os.path.exists(img_path):
        input(f"{img_path} already exists. Continue? >> ")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    assert 'acc_curve' in ckpt
    
    accs = ckpt['acc_curve']
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    plt.title(f'Acc curve plot', fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average acc", fontsize=12)

    plt.plot(np.arange(len(accs)), accs, linestyle='-')
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close('all')

def save_test_val_comparison_curve(ckpt_path):
    # ./active_learners/CIFAR10/active/seed_None/softmax_network/retrain/sequential/9000/active_coreset_norm_cosine/ckpt.pt
    img_path = ckpt_path[:ckpt_path.rfind('.')]+"_val_test_acc.png"
    if os.path.exists(img_path):
        input(f"{img_path} already exists. Continue? >> ")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    assert 'val_acc_curve' in ckpt
    assert 'test_acc_curve' in ckpt
    
    val_accs = ckpt['val_acc_curve']
    test_accs = ckpt['test_acc_curve']
    test_accs = [float(d['acc']) for d in test_accs]
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    plt.title(f'Acc curve plot', fontsize=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average acc", fontsize=12)
    plt.plot(np.arange(len(test_accs)), test_accs, linestyle='-', label="Test Accuracy")
    plt.plot(np.arange(len(val_accs)), val_accs, linestyle='-', label="Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', default='./active_learners/CIFAR10/active/seed_None/softmax_network/retrain/sequential') 
    args = parser.parse_args()
    import numpy as np
    b_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    method = "entropy"
    for b in b_list:
        ckpt_path_b = os.path.join(args.ckpt_path, f"{b}/active_{method}/ckpt.pt")
        ckpt = torch.load(ckpt_path_b, map_location=torch.device('cpu'))
        val_accs = np.array(ckpt['val_acc_curve'])
        # import pdb; pdb.set_trace()
        print(f"{b} query: Ckpt epoch is {np.argmax(val_accs)} with best val accuracy {np.max(val_accs)}")