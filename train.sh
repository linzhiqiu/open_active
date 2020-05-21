# Softmax Trainer
    # Regular setting
        python train.py CIFAR100 --training_method softmax_network --init_mode regular;
        python train.py CIFAR100 --training_method softmax_network --init_mode regular --dataset_rand_seed 1;
    # Fewer classes setting
        python train.py CIFAR100 --training_method softmax_network --init_mode fewer_class;
        python train.py CIFAR100 --training_method softmax_network --init_mode fewer_class --dataset_rand_seed 1;
    # Fewer samples setting
        python train.py CIFAR100 --training_method softmax_network --init_mode fewer_sample;
        python train.py CIFAR100 --training_method softmax_network --init_mode fewer_sample --dataset_rand_seed 1;
# Cosine Trainer
    # Regular setting
        python train.py CIFAR100 --training_method cosine_network --init_mode regular;
        python train.py CIFAR100 --training_method cosine_network --init_mode regular --dataset_rand_seed 1;
    # Fewer classes setting
        python train.py CIFAR100 --training_method cosine_network --init_mode fewer_class;
        python train.py CIFAR100 --training_method cosine_network --init_mode fewer_class --dataset_rand_seed 1;
    # Fewer samples setting
        python train.py CIFAR100 --training_method cosine_network --init_mode fewer_sample;
        python train.py CIFAR100 --training_method cosine_network --init_mode fewer_sample --dataset_rand_seed 1;


# If download dataset in local folder
# python train.py CIFAR100 --download_path ./data;
# python train.py CIFAR100 --download_path ./data --dataset_rand_seed 1;