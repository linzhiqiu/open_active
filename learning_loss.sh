# Plain CIFAR experiment
    # CIFAR10 + No active learning
    python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;
    # CIFAR100 + No active learning
    python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;

    # CIFAR10 + No active learning + only 10K examples
    python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;
    # CIFAR100 + No active learning + only 10K examples
    python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;

# CIFAR + 120 loss module epochs
    # CIFAR10 + loss learning
    python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;
    # CIFAR100 + loss learning
    python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;

    # CIFAR10 + loss learning + only 10K examples
    python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;
    # CIFAR100 + loss learning + only 10K examples
    python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;

# CIFAR + start from 1K examples to 10K

    # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K
    python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;


    # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling
    python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;


    # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K
    python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;


    # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling
    python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;