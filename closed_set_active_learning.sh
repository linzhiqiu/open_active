# Random query two query schemes
    # balanced init set
        python closed_set_active_learning.py CIFAR10 --training_method softmax_network --active_init_mode active --active_query_scheme sequential --active_train_mode retrain --query_method random --dataset_rand_seed None;
        python closed_set_active_learning.py CIFAR10 --training_method softmax_network --active_init_mode active --active_query_scheme independent --active_train_mode retrain --query_method random --dataset_rand_seed None;
    
    # random samples make sure sequential and independent have the same samples to start with
        python closed_set_active_learning.py CIFAR10 --training_method softmax_network --active_init_mode active --active_query_scheme sequential --active_train_mode retrain --query_method random --dataset_rand_seed None;
        python closed_set_active_learning.py CIFAR10 --training_method softmax_network --active_init_mode active --active_query_scheme independent --active_train_mode retrain --query_method random --dataset_rand_seed None;

    # suppose in phoenix
    --download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True