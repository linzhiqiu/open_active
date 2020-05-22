# Softmax Trainer
    # Regular setting - budget 10% = 3000 samples
        # Active learning : Entropy
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode regular --query_method entropy --budget 3000;
        # Active learning : Softmax
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode regular --query_method softmax --budget 3000;
        # Active learning : Random
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode regular --query_method random --budget 3000;
    # Fewer classes setting -- budget 3000
        # Active learning : Entropy
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode fewer_class --query_method entropy --budget 3000;
        # Active learning : Softmax
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode fewer_class --query_method softmax --budget 3000;
        # Active learning : Random
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode fewer_class --query_method random --budget 3000;
    # Fewer samples setting -- budget 3000
        # Active learning : Entropy
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode fewer_sample --query_method entropy --budget 3000;
        # Active learning : Softmax
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode fewer_sample --query_method softmax --budget 3000;
        # Active learning : Random
            # No random seed
                python train.py CIFAR100 --training_method softmax_network --init_mode fewer_sample --query_method random --budget 3000;
# Cosine Trainer
    # Regular setting -- budget 3000
        # Active learning : Entropy
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode regular --query_method entropy --budget 3000;
        # Active learning : Softmax
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode regular --query_method softmax --budget 3000;
        # Active learning : Random
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode regular --query_method random --budget 3000;
    # Fewer classes setting -- budget 3000
        # Active learning : Entropy
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode fewer_class --query_method entropy --budget 3000;
        # Active learning : Softmax
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode fewer_class --query_method softmax --budget 3000;
        # Active learning : Random
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode fewer_class --query_method random --budget 3000;
    # Fewer samples setting -- budget 3000
        # Active learning : Entropy
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode fewer_sample --query_method entropy --budget 3000;
        # Active learning : Softmax
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode fewer_sample --query_method softmax --budget 3000;
        # Active learning : Random
            # No random seed
                python train.py CIFAR100 --training_method cosine_network --init_mode fewer_sample --query_method random --budget 3000;

# If download dataset in local folder
# python train.py CIFAR100 --download_path ./data --budget 3000;
# python train.py CIFAR100 --download_path ./data --dataset_rand_seed 1 --budget 3000;