# Retrain mode
python analysis.py CIFAR100 --analysis_trainer softmax_network --train_mode retrain --budget_mode 1_5_10_20_50_100 --data_save_path ./datasets --data_download_path ./ --data_rand_seed None --trainer_save_dir ./trainers --analysis_save_dir ./analysis 
python analysis.py CIFAR100 --analysis_trainer cosine_network --train_mode retrain --budget_mode 1_5_10_20_50_100 --data_save_path ./datasets --data_download_path ./ --data_rand_seed None --trainer_save_dir ./trainers --analysis_save_dir ./analysis 


python analysis.py CIFAR10 --analysis_trainer softmax_network --train_mode retrain --budget_mode 1_5_10_20_50_100 --data_save_path ./datasets --data_download_path ./ --data_rand_seed None --trainer_save_dir ./trainers --analysis_save_dir ./analysis 
# python analysis.py CIFAR10 --analysis_trainer cosine_network --train_mode retrain --budget_mode 1_5_10_20_50_100 --data_save_path ./datasets --data_download_path ./ --data_rand_seed None --trainer_save_dir ./trainers --analysis_save_dir ./analysis 
