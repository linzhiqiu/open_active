
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;


python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.001 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
python train.py CIFAR100 --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.0001 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

# Softmax Network
    # Fixed threshold
        # Open Set accuracy stucks at 0
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

        # 
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.0001 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.0001 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;


        # 
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

        #
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;

    # Dynamic Threshold
        # 
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
        python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure random_query --pretrained CIFAR10;
        

# OSDN network
    # feature selection: Only correct
        # Threshold 0.01
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # Threshold 0.1
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # Threshold 0.5
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

    # feature selection: None correct then all
        # alpha_rank 10
            # Threshold 0.01
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

            # Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

            # Threshold 0.5
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # alpha_rank 40
            # Threshold 0.01
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_40 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_40 --weibull_tail_size fixed_20;

            # Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_40 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_40 --weibull_tail_size fixed_20;

            # Threshold 0.5
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_40 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_40 --weibull_tail_size fixed_20;

    # feature selection: all
        # Threshold 0.01
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # Threshold 0.1
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # Threshold 0.5
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;


# OSDN network Modified version without changing the seen class score
    # feature selection: None correct then all
        # alpha_rank 10
            # Threshold 0.01
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

            # Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

            # Threshold 0.5
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.5 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;