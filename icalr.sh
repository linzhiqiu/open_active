# Test without active learning see if prototype works
      # CIFAR10 All
      # CIFAR10 10K
      # CIFAR100 All
      # CIFAR100 10K
      # CIFAR10 10K Learning Loss.. Good accuracy
      # Need to check first round thresholds
      python train.py CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_10K --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_10K --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_10K --data_path ./data --trainer icalr_learning_loss --learning_loss_train_mode default --batch 128 --network_eval_threshold 0.0 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.05 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 

# Test without active learning see if prototype works for open set
      # CIFAR10 All 5 open class: Test 85.2
      # CIFAR10 10K 5 open class: Train 100, Test 95.4
      # CIFAR100 All 50 open class: Test 78.1
      # CIFAR100 10K 50 open class: Test 55.7
      # CIFAR10 10K Learning Loss.. Good accuracy 50 open class
      # Need to check first round thresholds
      python train.py CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_5_open_classes --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_5K_5_open_classes --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_5K_50_open_classes --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_50_open_classes --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode no_learning_5K_5_open_classes --data_path ./data --trainer icalr_learning_loss --learning_loss_train_mode default --batch 128 --network_eval_threshold 0.0 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.05 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 

      # Old open set setting
            # Softmax not pretrained
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer network --batch 128 --network_eval_threshold 0.1 --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident;
            # Binary Softmax not pretrained
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer binary_softmax --batch 128 --network_eval_threshold 0.1 --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident;
            # Entropy not pretrained
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer network --threshold_metric entropy --network_eval_mode dynamic_threshold --batch 128 --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident;
            # ICALR Softmax not pretrained
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.1 --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident;
            # ICALR Entropy not pretrained
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr --threshold_metric entropy --batch 128 --network_eval_mode dynamic_threshold --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident;
            # ICALR Binary Softmax (fixed variance) not pretrained
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_binary_softmax --icalr_binary_softmax_train_mode fixed_variance --batch 128 --network_eval_threshold 0.1 --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident;
            # Sigmoid + mean 
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --sigmoid_train_mode mean;
            # Sigmoid + sum
            python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 128 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --sigmoid_train_mode sum;
            # OSDN
                  # EUCOS + threshold 0.1
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 128 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                  # EUCOS + pseudo threshold + metric weighted alpha rank 2
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
                  # EUCOS + pseudo threshold + metric average alpha rank 5
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
            # OSDN Modified
                  # EUCOS + threshold 0.1
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 128 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                  # EUCOS + pseudo threshold + weighted metric alpha rank 2
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
                  # EUCOS + pseudo threshold + average metric alpha rank 5
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
            # ICALR OSDN
                  # EUCOS + threshold 0.1
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn --batch 128 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                  # EUCOS + pseudo threshold + metric weighted alpha rank 2
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
                  # EUCOS + pseudo threshold + metric average alpha rank 5
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
            # ICALR OSDN Modified
                  # EUCOS + threshold 0.1
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_modified --batch 128 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                  # EUCOS + pseudo threshold + weighted metric alpha rank 2
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_modified --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
                  # EUCOS + pseudo threshold + average metric alpha rank 5
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_modified --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
            # ICALR OSDN Negative Open set score
                  # EUCOS + threshold 0.1
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_neg --batch 128 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                  # EUCOS + pseudo threshold + metric weighted alpha rank 2
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_neg --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
                  # EUCOS + pseudo threshold + metric average alpha rank 5
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_neg --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
            # ICALR OSDN Modified Negative Open set score
                  # EUCOS + threshold 0.1
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_modified_neg --batch 128 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
                  # EUCOS + pseudo threshold + weighted metric alpha rank 2
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_modified_neg --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
                  # EUCOS + pseudo threshold + average metric alpha rank 5
                  python train.py CIFAR100 --verbose False --log_first_round_thresholds True --log_test_accuracy False --init_mode open_set_leave_one_out_new --writer False --save_ckpt False --data_path ./data --trainer icalr_osdn_modified_neg --batch 128 --mav_features_selection none_correct_then_all --arch ResNet18 --lr 0.1 --epochs 200 --lr_decay_step 160 --uncertainty_measure least_confident --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;

# Test 40 class 25 example scenario
      # 1: Overfitting to train example..
      # 2: Train 22, Test 15 on 40 classes
      # 3: Train 100, Test 23 on 40 classes
      # 4: Running
      # 5: Running 
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_1 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_1 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 5 --epochs 10 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_1 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 30 --epochs 60 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_1 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 15 --epochs 25 --wd 5e-4 --uncertainty_measure random_query;
      python train.py CIFAR100 --pretrained CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_1 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet50 --lr 0.1 --lr_decay_step 30 --epochs 60 --wd 5e-4 --uncertainty_measure random_query;

# Test 100 class 20 example scenario
      # 1: Train acc 100, Test acc 20
      # 2: Train acc 0.162, Test acc 10
      # 3: Train acc 99, Test acc 18.3
      # 4: Train acc 40, Test acc 15
      # 5: Pretrained ResNet50, Train 100 Test 36
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_3 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_3 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 5 --epochs 10 --wd 5e-4 --uncertainty_measure random_query;
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_3 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 30 --epochs 60 --wd 5e-4 --uncertainty_measure random_query;
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_3 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 15 --epochs 25 --wd 5e-4 --uncertainty_measure random_query;
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --pretrained CIFAR10 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode few_shot_3 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet50 --lr 0.1 --lr_decay_step 30 --epochs 60 --wd 5e-4 --uncertainty_measure random_query;

# Test 100 class 50/100/150 example scenario
      # 1: Running
      # 2: Running
      # 3: Running
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode many_shot_1 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode many_shot_2 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;
      CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --log_first_round_thresholds True --save_ckpt False --verbose True --init_mode many_shot_3 --data_path ./data --trainer icalr --max_rounds 1 --budget 0 --batch 128 --network_eval_threshold 0.00 --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --wd 5e-4 --uncertainty_measure random_query;


# CIFAR100 + Softmax Uncertainty Threshold 2
      python train.py CIFAR100 --max_rounds 400 --budget 20 --icalr_mode default --icalr_retrain_threshold 40 --icalr_strategy proto  --save_ckpt False --verbose True --log_everything True --init_mode open_active_1 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 400 --budget 20 --icalr_mode default --icalr_retrain_threshold 40 --icalr_strategy proto  --save_ckpt False --verbose True --log_everything True --init_mode open_active_1 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 400 --budget 20 --icalr_mode default --icalr_retrain_threshold 40 --icalr_strategy proto  --save_ckpt False --verbose True --log_everything True --init_mode open_active_1 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 400 --budget 20 --icalr_mode default --icalr_retrain_threshold 40 --icalr_strategy proto  --save_ckpt False --verbose True --log_everything True --init_mode open_active_1 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 400 --budget 20 --icalr_mode default --icalr_retrain_threshold 40 --icalr_strategy proto  --save_ckpt False --verbose True --log_everything True --init_mode open_active_1 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 400 --budget 20 --icalr_mode default --icalr_retrain_threshold 40 --icalr_strategy proto  --save_ckpt False --verbose True --log_everything True --init_mode open_active_1 --learning_loss_stop_epoch 10 --data_path ./data --trainer icalr_learning_loss --learning_loss_train_mode default --batch 128 --network_eval_threshold 0.02 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.05 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;

      python train.py CIFAR100 --max_rounds 250 --budget 20 --icalr_mode default --icalr_retrain_threshold 25 --icalr_strategy proto --save_ckpt False --verbose True --log_everything True --init_mode open_active_2 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure random_query --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 250 --budget 20 --icalr_mode default --icalr_retrain_threshold 25 --icalr_strategy proto --save_ckpt False --verbose True --log_everything True --init_mode open_active_2 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure most_confident --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 250 --budget 20 --icalr_mode default --icalr_retrain_threshold 25 --icalr_strategy proto --save_ckpt False --verbose True --log_everything True --init_mode open_active_2 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure least_confident --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 250 --budget 20 --icalr_mode default --icalr_retrain_threshold 25 --icalr_strategy proto --save_ckpt False --verbose True --log_everything True --init_mode open_active_2 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --uncertainty_measure entropy --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 250 --budget 20 --icalr_mode default --icalr_retrain_threshold 25 --icalr_strategy proto --save_ckpt False --verbose True --log_everything True --init_mode open_active_2 --learning_loss_stop_epoch 0 --data_path ./data --trainer icalr --batch 128 --network_eval_threshold 0.02 --active_random_sampling none --arch ResNet18 --lr 0.1 --lr_decay_step 160 --epochs 200 --label_picker coreset_measure --coreset_measure k_center_greedy --coreset_feature before_fc --wd 5e-4 ;
      python train.py CIFAR100 --max_rounds 250 --budget 20 --icalr_mode default --icalr_retrain_threshold 25 --icalr_strategy proto --save_ckpt False --verbose True --log_everything True --init_mode open_active_2 --learning_loss_stop_epoch 20 --data_path ./data --trainer icalr_learning_loss --learning_loss_train_mode default --batch 128 --network_eval_threshold 0.02 --active_random_sampling fixed_10K --arch ResNet18 --lr 0.05 --lr_decay_step 160 --epochs 200 --uncertainty_measure lowest_loss --wd 5e-4 ;

