# # Plain CIFAR experiment
#     # CIFAR10 + No active learning: TEST 91.3%
#     python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
#     # CIFAR100 + No active learning: TEST 74.6%
#     python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;

#     # CIFAR10 + No active learning + only 10K examples: TEST 84.1%
#     python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
#     # CIFAR100 + No active learning + only 10K examples: TEST 53.1%
#     python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;

# # CIFAR + 120 loss module epochsython train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure learning_loss --wd 5e-4 ;
#     # CIFAR10 + loss learning: TEST 90.9% Best Loss pred Acc is 83% before 120 epoch. Then it degrades to 53%
#     python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
#     # Rerunning.. CIFAR100 + loss learning: TEST 70.6% Best Loss pred Acc is 71% before 120 epochs (while train acc stuck at 56%). Then degrades to 50%.
#     python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;

#     # CIFAR10 + loss learning + only 10K examples: TEST 83.4% Best loss pred acc is 84% before 120 epochs (while train acc stucks at 75%). 
#     python train.py CIFAR10  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
#     # Rerunning.. CIFAR100 + loss learning + only 10K examples: TEST 49.8% Best loss pred acc is 67% before 120 epochs (while train acc stuck at 76%).
#     python train.py CIFAR100  --save_ckpt False --verbose True --init_mode no_learning_10K --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;

# CIFAR + start from 1K examples to 10K

    # # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
    # python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
    # python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
    # python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
    # python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [60-120]
    # python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_start_epoch 60 --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [60-120]
    # python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_start_epoch 60 --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [60-120]
    # python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_start_epoch 60 --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;


    # # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [60-120]
    # python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_start_epoch 60 --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;

    # Start with fixed number of examples
        # No random sampling
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
            


            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
        
        # Random sampling 10K (TODO all)
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
            


            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
            
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-0]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-0]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;

            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-0]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-0]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            
    # Start with random samples
        # No random sampling
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
            


            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
        
        # Random sampling 10K (TODO all)
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
            


            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random query!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + least_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + most_confident!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + entropy!
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set before fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + core set after fc
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature after_fc --wd 5e-4 ;
            
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;

            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-0]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-0]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-0]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-0]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure highest_loss --wd 5e-4 ;

            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-0]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-0]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR10 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
            python train.py CIFAR10  --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            

            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-0]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-0]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 0 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + random_sampling 10K + prop epoch [0-120]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;
            # CIFAR100 + Softmax Uncertainty Threshold 0.00 + No random_sampling + prop epoch [0-120]
            python train.py CIFAR100 --save_ckpt False --verbose True --init_mode learning_loss_start_random --learning_loss_stop_epoch 120 --data_path ./data --trainer network_learning_loss --learning_loss_train_mode default --max_rounds 10 --budget 1000 --batch 128 --network_eval_threshold 0.00 --active_random_sampling none  --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;

            
