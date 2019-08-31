# CIFAR100 10 closed + 10 open
    # ImageLevelGAN
        # Single player GAN
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player single --gan_mode ImageLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
        # Multiple player GAN
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player multiple --gan_mode ImageLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
        # Multiple player GAN + highest Discriminator
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player multiple --gan_mode ImageLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --gan_multi highest;
        # Multiple player GAN + lowest Discriminator
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player multiple --gan_mode ImageLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --gan_multi lowest;
        # Background player GAN
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background --gan_mode ImageLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
            # 0.3 open set, 0.6 closed set: python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background --gan_mode ImageLevelGAN --gan_setup 20_epochs --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --save_gan_output True;
            # All treated as open set: python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background --gan_mode ImageLevelGAN --gan_setup 20_epochs --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --save_gan_output True --gan_multi lowest; 
            # Almost all are open set: python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background --gan_mode ImageLevelGAN --gan_setup 20_epochs --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 10 --save_gan_output True --gan_multi highest;
        # Background + noise player GAN + all open set
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background_noise --gan_mode ImageLevelGAN --gan_setup 100_epochs --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --save_gan_output True;
    # FeatureLevelGAN
        # Single player GAN + open 0.45, seen 0.4
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player single --gan_mode FeatureLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
        # Multiple player GAN + open 0, seen 0.9,
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player multiple --gan_mode FeatureLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
        # Multiple player GAN + highest Discriminator + open 0.5, closed 0.5
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player multiple --gan_mode FeatureLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --gan_multi highest;
        # Multiple player GAN + lowest Discriminator
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player multiple --gan_mode FeatureLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20 --gan_multi lowest;
        # Background player GAN + open set 0.16, closed 0.83
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background --gan_mode FeatureLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
        # Background + noise player GAN + open set 0.2, closed 
            python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer gan --gan_player background_noise --gan_mode FeatureLevelGAN --gan_setup standard --batch 32 --arch ResNet50 --lr 0.1 --uncertainty_measure least_confident --pretrained CIFAR10 --epochs 20;
    # Cluster Level after_fc
        # weighted metric
            #EUCOS + Pseudo open 1 classes
                # LR 0.1 + Loss NaN
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
            # LR 0.001 train acc 0.1
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.000001
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.00000001 train acc 0.23, loss 2.2
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.00000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
            # EU
                # LR 0.1  + train acc 0.584, loss 2.3
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.01 + train acc 0.46, loss 2.3
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.001 + train acc 0.42, loss 2.3
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500;
            # COS
                # LR 0.1 + train acc 0.96, loss 1.49
                python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.01 + train acc 0.86, loss 1.55
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.001 + train acc 0.89, loss 1.55
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
        # 7_3 metric for pseudo open set optimal threshold
            #EUCOS + Pseudo open 1 classes
                # LR 0.001 - train acc 0.18, loss 20
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                # LR 0.000001 - train acc 0.457, loss 1.9
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
            # EU
                # Div by 200
                    # LR 0.1 - train acc 0.56
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                    # LR 0.01 - train acc 0.45
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                # Div by 1.
                    # LR 0.1
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3 --div_eu 1.0;
                    # LR 0.01
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3 --div_eu 1.0;
            # COS
                # LR 0.1 + train acc 98
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                # LR 0.001 + train acc 80
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level after_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
    # Cluster Level before_fc
        # weighted metric + 1 pseudo open class
            #EUCOS + Pseudo open 1 classes
                # LR 0.1 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.001
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.000001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.00000001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.00000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
            # EU + Pseudo open 1 classes
                # LR 0.1 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.01 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500;
            # COS + Pseudo open 1 classes
                # LR 0.1 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.01 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
                # LR 0.001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500;
        # weighted metric + 5 open class
            #EUCOS 
                # LR 0.1 + div_eu 5.
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
                # LR 0.000001 + div_eu 5.
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
                # LR 0.000001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # LR 0.00000001 + train acc 0.3178
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.00000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # EU 
                # LR 0.1 + train acc 0.7324, threshold 0.548. open 0.29, closed 0.18
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # LR 0.1 + eu div by 5. + train 98, open 0.91, closed 0.09
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
                # LR 0.1 + eu div by 10. + train acc 96, open 1, closed 0
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 10.;
                # LR 0.1 + eu div by 20. + train acc 84, open 1, closed 0
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 20.;
                # LR 0.01 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # COS 
                # LR 1.0 + train acc 0.8764, open 0.84, closed 0.10
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 1.0 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # LR 0.5
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.5 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # LR 0.1 + train acc 0.9878, threshold 0.25, open 1, closed 0
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # LR 0.05 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.05 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
                # LR 0.01 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
        # average metric
            #EUCOS + Pseudo open 1 classes
                # LR 0.00000001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.00000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
            # EU + Pseudo open 1 classes
                # LR 0.1 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
            # COS + Pseudo open 1 classes
                # LR 0.1 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
        # 7_3 metric for pseudo open set optimal threshold
            #EUCOS + Pseudo open 1 classes
                # LR 0.001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                # LR 0.000001
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
            # EU
                # Div by 200
                    # LR 0.1
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                    # LR 0.01 
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                # Div by 1.
                    # LR 0.1
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3 --div_eu 1.0;
                    # LR 0.01
                        python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3 --div_eu 1.0;
            # COS
                # LR 0.1
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
                # LR 0.001 
                    python train.py CIFAR100 --verbose True --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
    # Softmax Uncertainty 
        # Threshold 0.1
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
        # Pseudo open set 1 + weighted metric
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 1 --pseudo_open_set_rounds 500;   
        # Pseudo open set 1 + average metric + threshold 0.883, train acc 0.98, open 0.44, closed 0.82
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
        # Pseudo open set 5 + average metric + threshold 1.0, open 0.98, seen 0.29
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
        # Pseudo open set 1 + 7_3 metric + threshold 0.25, closed 0.91, open 0
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;   
        # Pseudo open set 5 + 7_3 metric + threshold 0.31, closed 0.91, open 0
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;   
        # Pseudo open set 5 + weighted metric + 0.55 threshold, closed 0.87, open 0.09
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted;
        # Pseudo open set 5 + weighted metric + same network + 
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --pseudo_same_network True;   
        # Pseudo open set 4 + weighted metric + same network + open 0, closed 0.9, threshold 0.3 and fails all open example
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 4 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --pseudo_same_network True;   
        # Pseudo open set 6 + weighted metric + same network + open 0.99, closed 0.3 threshold 1.0
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 6 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --pseudo_same_network True;   
        # Pseudo open set 4 + weighted metric + no same network + open 0, closed 0.91, threshold 0.31
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 4 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --pseudo_same_network False;   
        # Pseudo open set 6 + weighted metric + no same network + open 0.99, closed 0.36, threshold 1.0
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 6 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --pseudo_same_network False;   
    # Entropy Uncertainty
        # Pseudo open set 1 + weighted metric 
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 1 --pseudo_open_set_rounds 500;
        # Pseudo open set 1 + average metric + threshold -1.27, closed 0.9, open 0
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
        # Pseudo open set 1 + 7_3 metric, open set acc is 0, threshold -1.86
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 1 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
        # Pseudo open set 5 + weighted metric + threshold -.82, closed 91, open 0.2
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
        # Pseudo open set 5 + weighted metric + same network
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_same_network True;
        # Pseudo open set 5 + average metric + threshold -0.8, open set 0.2, seen class 0.90
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric average;
        # Pseudo open set 5 + 7_3 metric, open set acc is 0 + threshold -1.57
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric 7_3;
        # Pseudo open set 4 + weighted metric + same network + open 0 closed 0.9 threshold -1.5
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 4 --pseudo_open_set_rounds 500 --pseudo_same_network True;
        # Pseudo open set 4 + weighted metric + no same network + open 0 closed 0.89 threshold -1.6
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 4 --pseudo_open_set_rounds 500 --pseudo_same_network False;
        # Pseudo open set 6 + weighted metric + same network + open 1 closed 0 threshold -4.3
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 6 --pseudo_open_set_rounds 500 --pseudo_same_network True;
        # Pseudo open set 6 + weighted metric + no same network + open 0.99 closed 0.12 threshold -3.5
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 6 --pseudo_open_set_rounds 500 --pseudo_same_network False;
    # OSDN
        # EUCOS + threshold 0.1
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
        # EUCOS + pseudo threshold + metric weighted alpha rank 2
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
        # EUCOS + pseudo threshold + metric average alpha rank 5
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
    # OSDN Modified
        # EUCOS + threshold 0.1
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
        # EUCOS + pseudo threshold + weighted metric alpha rank 2
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
        # EUCOS + pseudo threshold + average metric alpha rank 5
            python train.py CIFAR100 --verbose False --log_first_round True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;

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
            # Softmax + round 500 + round 200 achieve balanced open set and closed set accuracy, but open set (50 - 80) better than closed set (around 28)
                python train.py CIFAR100 --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 5 --pseudo_open_set_rounds 500;
            # Entropy + max pseudo round 500 + round 200 achieve a balanced closed + open set accuracy around 40%.
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
            # Entropy + pseudo round 500 + round 122 higher open set accuracy (50-70) than closed set acc (around 23)
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