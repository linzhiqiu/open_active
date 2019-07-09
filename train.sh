# Comparison between different methods
    # No imbalanced weight
        # Least confident
            # Uncertainty Threshold 0.1
                # Softmax
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
                # Dynamic Softmax Threshold
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
                # OSDN (None correct then select all features) Alpha rank 10, Weibull size 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                # OSDN Modified (None correct then select all features) Alpha rank 10, Weibull size 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # Most confident
            # Uncertainty Threshold 0.1
                # Softmax
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
                # Dynamic Softmax Threshold
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
                # OSDN (None correct then select all features) Alpha rank 10, Weibull size 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                # OSDN Modified (None correct then select all features) Alpha rank 10, Weibull sizsee 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

    # Pseudo-openset class and no imbalanced weight
        # Least confident
            # Softmax
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 1;
            # Entropy
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 1;
            # OSDN (None correct then select all features)
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;
            # OSDN Modified (None correct then select all features)
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;

        # Most confident
            # Softmax
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 1;
            # Entropy
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 1;
            # OSDN (None correct then select all features)
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;
            # OSDN Modified (None correct then select all features)
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;


    # With class imbalanced weight
        # Least confident
            # Uncertainty Threshold 0.1
                # Softmax
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --class_weight class_imbalanced;
                # Dynamic Softmax Threshold
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --class_weight class_imbalanced;
                # OSDN (None correct then select all features) Alpha rank 10, Weibull size 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20 --class_weight class_imbalanced;
                # OSDN Modified (None correct then select all features) Alpha rank 10, Weibull size 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20 --class_weight class_imbalanced;


        # Most confident
            # Uncertainty Threshold 0.1
                # Softmax
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --class_weight class_imbalanced;
                # Dynamic Softmax Threshold
                    python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --class_weight class_imbalanced;
                # OSDN (None correct then select all features) Alpha rank 10, Weibull size 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20 --class_weight class_imbalanced;
                # OSDN Modified (None correct then select all features) Alpha rank 10, Weibull sizsee 20
                    python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20 --class_weight class_imbalanced;

    # Random Query
        # Uncertainty Threshold 0.1
            # Dynamic Softmax Threshold
                 



# Archive: Below are scripts with old open active setup
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
        # alpha_rank 5
            # Threshold 0.01
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_5 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.01 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_5 --weibull_tail_size fixed_20;

            # Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_5 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_5 --weibull_tail_size fixed_20;

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
        # alpha_rank 5
            # Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_5 --weibull_tail_size fixed_20;
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_5 --weibull_tail_size fixed_20;

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