# Cluster Level before_fc
# weighted metric + 5 open class
#EUCOS 
# LR 0.1 + div_eu 5.
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
# LR 0.000001 + div_eu 5.
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
# LR 0.000001 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.00000001
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.00000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.00000001 + div_eu 5.
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.00000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
# LR 0.000000001 + div_eu 5.
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.000000001 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
# EU 
# LR 0.1 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.1 + eu div by 5. 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 5.;
# LR 0.1 + eu div by 10. 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 10.;
# LR 0.1 + eu div by 20. 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500 --div_eu 20.;
# LR 0.01 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eu --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# COS 
# LR 1.0 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 1.0 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.5
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.5 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.1
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.05 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.05 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;
# LR 0.01 
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer cluster --cluster_level before_fc  --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric gaussian --clustering rbf_train --rbf_gamma 1.0 --arch ResNet50 --lr 0.01 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric cos --pseudo_open_set 5 --pseudo_open_set_rounds 500;

# Sigmoid Uncertainty
# Threshold 0.5 + mean
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 32 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --sigmoid_train_mode mean;
# Threshold 0.5 + sum
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 32 --network_eval_threshold 0.5 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --sigmoid_train_mode sum;
# Pseudo open set 1 + weighted metric + mean
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 1 --pseudo_open_set_rounds 500 --sigmoid_train_mode mean;   
# Pseudo open set 1 + weighted metric + sum
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 1 --pseudo_open_set_rounds 500 --sigmoid_train_mode sum;   
# Pseudo open set 5 + weighted metric + mean
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --sigmoid_train_mode mean;
# Pseudo open set 5 + weighted metric + sum
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer sigmoid --batch 32 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --network_eval_mode pseuopen_threshold --pseudo_open_set 5 --pseudo_open_set_rounds 500 --pseudo_open_set_metric weighted --sigmoid_train_mode sum;

# Softmax Uncertainty 
# Threshold 0.1
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_threshold 0.1 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10;
# Entropy Uncertainty
# Pseudo open set 1 + weighted metric 
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer network --batch 32 --network_eval_mode pseuopen_threshold --threshold_metric entropy --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --pseudo_open_set 1 --pseudo_open_set_rounds 500;
# OSDN
# EUCOS + threshold 0.1
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
# EUCOS + pseudo threshold + metric weighted alpha rank 2
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
# EUCOS + pseudo threshold + metric average alpha rank 5
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;
# OSDN Modified
# EUCOS + threshold 0.1
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --osdn_eval_threshold 0.1 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos --alpha_rank fixed_10 --weibull_tail_size fixed_20;
# EUCOS + pseudo threshold + weighted metric alpha rank 2
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric weighted;
# EUCOS + pseudo threshold + average metric alpha rank 5
python train.py CIFAR100 --verbose False --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer osdn_modified --batch 32 --mav_features_selection none_correct_then_all --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --distance_metric eucos  --pseudo_open_set 1 --pseudo_open_set_rounds 500 --openmax_meta_learn open_set --pseudo_open_set_metric average;

# C2AE Uncertainty
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode default;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode default_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode default_bce;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_mse;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_bce;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim sgd --c2ae_alpha 0.9 --c2ae_train_mode default;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.1 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim sgd --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.8 --c2ae_train_mode default;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.8 --c2ae_train_mode a_minus_1;

# C2AE debug purpose with alpha 1.0
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_bce;

CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_dcgan;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_not_frozen_dcgan;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_simple_autoencoder;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_simple_autoencoder_bce;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_not_frozen;

python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_simple_autoencoder_bce;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_simple_autoencoder_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_simple_autoencoder;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode debug_simple_autoencoder_bce;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode debug_simple_autoencoder_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.9 --c2ae_train_mode debug_simple_autoencoder;

python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.8 --c2ae_train_mode debug_simple_autoencoder_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.8 --c2ae_train_mode debug_simple_autoencoder;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.7 --c2ae_train_mode debug_simple_autoencoder_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.7 --c2ae_train_mode debug_simple_autoencoder;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.6 --c2ae_train_mode debug_simple_autoencoder_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 50 --uncertainty_measure least_confident --pretrained CIFAR10 --optim adam --c2ae_alpha 0.6 --c2ae_train_mode debug_simple_autoencoder;

# C2AE test their classifier32
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_dcgan;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_dcgan_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_not_frozen_dcgan_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 1.0 --c2ae_train_mode debug_no_label_not_frozen_dcgan;

CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan_mse;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan_mse;
CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan;


python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan;

    python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_instancenorm_dcgan_mse --c2ae_train_in_eval_mode False --c2ae_instancenorm_affine False;
    python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_instancenorm_dcgan --c2ae_train_in_eval_mode False --c2ae_instancenorm_affine False;
    CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_instancenorm_dcgan_mse --c2ae_train_in_eval_mode False --c2ae_instancenorm_affine False;
    CUDA_VISIBLE_DEVICES=1 python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_instancenorm_dcgan --c2ae_train_in_eval_mode False --c2ae_instancenorm_affine False;

    python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_instancenorm_affine_dcgan_mse --c2ae_train_in_eval_mode False;
    python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_instancenorm_affine_dcgan --c2ae_train_in_eval_mode False;
    python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_instancenorm_affine_dcgan_mse --c2ae_train_in_eval_mode False;
    python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32_instancenorm --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_instancenorm_affine_dcgan --c2ae_train_in_eval_mode False;


python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan_mse_not_frozen;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan_not_frozen;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan_mse_not_frozen;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan_not_frozen;

python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan_mse_not_frozen;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode a_minus_1_dcgan_not_frozen;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan_mse_not_frozen;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 30 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode a_minus_1_dcgan_not_frozen;


python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode UNet;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.9 --c2ae_train_mode UNet_mse;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode UNet;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch classifier32 --lr 0.0003 --epochs 150 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.7 --c2ae_train_mode UNet_mse;

python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 25 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.8 --c2ae_train_mode UNet;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 25 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.85 --c2ae_train_mode UNet;
python train.py CIFAR100 --verbose True --log_first_round_thresholds True --init_mode open_set_leave_one_out --writer False --save_ckpt False --data_path ./data --trainer c2ae --batch 64 --arch ResNet50 --lr 0.0003 --epochs 25 --uncertainty_measure least_confident --optim adam --c2ae_alpha 0.8 --c2ae_train_mode UNet_mse;

'debug_simple_autoencoder_bce', 'debug_simple_autoencoder_mse', 'debug_simple_autoencoder'