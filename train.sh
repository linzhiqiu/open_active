# CIFAR100 10 closed + 10 open
    # Cluster
        # EUCOS + Pseudo open 5 classes
            # LR 0.1
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # LR 0.001
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # LR 0.000001
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
        # EU
            # LR 0.1
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # LR 0.01
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
        # COS
            # LR 0.1
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # LR 0.01
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # LR 0.001
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
    # Softmax Uncertainty 
        # Threshold 0.1
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
        # Pseudo open set 5
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500;   
    # Entropy Uncertainty
        # Pseudo open set 5
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
    # OSDN
        # EUCOS + threshold 0.1
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
        # EUCOS + pseudo threshold
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set;
    # OSDN Modified
        # EUCOS + threshold 0.1
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
        # EUCOS + pseudo threshold
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set;

# Comparison between different methods
    # No imbalanced weight
        # Least confident
            # Softmax Uncertainty Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
            # Dynamic Softmax Threshold
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
            # Dynamic Entropy Threshold. Result: Bad. Everything is seen class.
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
            # OSDN (None correct then select all features) Alpha rank 10, Weibull size 20
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            # OSDN Modified (None correct then select all features) Alpha rank 10, Weibull size 20
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

        # Most confident
            # Softmax Uncertainty Threshold 0.1
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
            # Dynamic Softmax Threshold
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
            # Dynamic Entropy Threshold. Result: Bad. Everything is open class.
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode dynamic_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10;
            # OSDN (None correct then select all features) Alpha rank 10, Weibull size 20
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
            # OSDN Modified (None correct then select all features) Alpha rank 10, Weibull sizsee 20
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;

    # Pseudo-openset class and no imbalanced weight
        # Least confident
            # Softmax + round 500
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # Entropy + max pseudo round 500
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # OSDN (None correct then select all features) + round 1
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;
            # OSDN Modified (None correct then select all features) + round 1
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;
            # OSDN Based New pseudo open setting + average of closed and open metric
                # Default hyper
                    # OSDN (None correct then select all features) + round 500. Result Not promising. Seen acc < 0.02
                        python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn default --pseudo_open_set_metric average;
                    # OSDN Modified (None correct then select all features) + round 500. Seen acc < 0.09
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn default --pseudo_open_set_metric average;
                # Advanced hyper 
                    # OSDN (None correct then select all features) + round 500. Result: Not promising. All examples degrade to open set. Seen class acc < 0.05
                        python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric average;
                    # OSDN Modified (None correct then select all features) + round 500. Result: Not promising. All examples degrade to open set. Seen class acc < 0.05
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric average;
            # OSDN Based New pseudo open setting + weighted metric
                # Advanced hyper 
                    # OSDN (None correct then select all features) + round 500. Not successful. Result: Not promising. All examples degrade to open set. Seen class acc < 0.02. Alpha always pick 2.
                        python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric weighted;
                    # OSDN Modified (None correct then select all features) + round 500. Result: Not promising. All examples degrade to open set. Seen class acc < 0.07. Alpha always pick 2.
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric weighted;
                # More alpha
                    # OSDN Modified (None correct then select all features) + round 500. Still degrade to alpha rank 2
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn morealpha --pseudo_open_set_metric weighted;
            # Cluster RBF Train + Pseudo open class
                # EUCOS LR 0.001
                    python train.py CIFAR100 --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # EUCOS LR 0.000001
                    python train.py CIFAR100 --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # EU LR 0.1
                    python train.py CIFAR100 --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # COS LR 0.1
                    python train.py CIFAR100 --data_path ./data --trainer cluster  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric softmax --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;


        # Most confident
            # Softmax + round 500
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # Entropy + pseudo round 1
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 1;
            # Entropy + pseudo round 5
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 5;
            # Entropy + pseudo round 500
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # OSDN (None correct then select all features) + round 1
                python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;
            # OSDN Modified (None correct then select all features) + round 1
                python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 1 --openmax_meta_learn default;
            # OSDN Based New pseudo open setting + average of closed and open metric
                # Default hyper
                    # OSDN (None correct then select all features) + round 500. Not promising. Seen class < 0.06
                        python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn default --pseudo_open_set_metric average;
                    # OSDN Modified (None correct then select all features) + round 500. Seen class < 0.08
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn default --pseudo_open_set_metric average;
                # Advanced hyper 
                    # OSDN (None correct then select all features) + round 500. Result: Not promising. All examples degrade to open set. Seen class acc < 0.07
                        python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric average;
                    # OSDN Modified (None correct then select all features) + round 500. Result: Somewhat promising. All examples degrade to open set. Seen class acc < 0.10. But like the others, the alpha rank is fixed at 2.
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric average;
            # OSDN Based New pseudo open setting + weighted metric
                # Advanced hyper 
                    # OSDN (None correct then select all features) + round 500. Result: Not promising. All examples degrade to open set. Seen class acc < 0.06. Alpha always pick 2.
                        python train.py CIFAR100 --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric weighted;
                    # OSDN Modified (None correct then select all features) + round 500. Result: Not promising. All examples degrade to open set. Seen class acc < 0.08. Alpha always pick 2.
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn advanced --pseudo_open_set_metric weighted;
                # More alpha hyper 
                    # OSDN Modified (None correct then select all features) + round 500. Still degrade to alpha rank 2
                        python train.py CIFAR100 --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure most_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 5 --pseudo_open_set_rounds 500 --openmax_meta_learn morealpha --pseudo_open_set_metric weighted;

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