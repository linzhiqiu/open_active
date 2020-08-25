# CIFAR10 Stratified
    # Coreset Cosine (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Coreset (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Random (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Softmax (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Entropy (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # ULDR (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # ULDR Cosine (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed None --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 10 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 100 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR10 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 2000 --data_download_path ./ --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True







#CIFAR 100 Stratified
    # Coreset Cosine (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset_norm_cosine --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Coreset (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method coreset --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method coreset --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Random (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method random --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method random --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Softmax (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method softmax --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method softmax --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # Entropy (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method entropy --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method entropy --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # ULDR (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

    # ULDR Cosine (Seed: None, 1, 10, 100, 1000, 2000)
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed None --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 10 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 100 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 1000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True

        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme sequential --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets  --active_save_dir ./active_learners --verbose True
        python start_active_learning.py CIFAR100 --training_method softmax_network --data_config active --active_query_scheme independent --train_mode retrain --query_method uldr_norm_cosine --data_rand_seed 2000 --data_download_path /scratch --active_save_path ./active_datasets --active_save_dir ./active_learners --verbose True


