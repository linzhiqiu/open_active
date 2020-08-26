# Retrain mode
python open_set_active_analysis.py CIFAR10 --budget_mode 1_5_10_20_50_100 --analysis_save_dir ./open_active_analysis
python open_set_active_analysis.py CIFAR100 --budget_mode 1_5_10_20_50_100 --analysis_save_dir ./open_active_analysis
