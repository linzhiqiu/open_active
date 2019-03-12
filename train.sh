python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;


python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.001 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.0001 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;