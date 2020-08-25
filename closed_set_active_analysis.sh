python closed_set_active_analysis.py CIFAR10 --active_save_path ./active_datasets --active_analysis_save_dir ./active_analysis --active_analysis_trainer softmax_network --active_save_dir ./active_learners --train_mode retrain
python closed_set_active_analysis.py CIFAR100 --active_save_path ./active_datasets --active_analysis_save_dir ./active_analysis --active_analysis_trainer softmax_network --active_save_dir ./active_learners --train_mode retrain


python closed_set_active_analysis.py CIFAR10 --active_save_path ./active_datasets_val_set --active_analysis_save_dir ./active_analysis_val_set --active_analysis_trainer softmax_network --active_save_dir ./active_learners_val_set --train_mode retrain
python closed_set_active_analysis.py CIFAR100 --active_save_path ./active_datasets_val_set --active_analysis_save_dir ./active_analysis_val_set --active_analysis_trainer softmax_network --active_save_dir ./active_learners_val_set --train_mode retrain