python analysis.py CIFAR100 --analysis_trainer softmax_network --budget_mode 1_5_10_20_50_100
python analysis.py CIFAR100 --analysis_trainer cosine_network --budget_mode 1_5_10_20_50_100



# No finetune mode
python analysis.py CIFAR100 --analysis_trainer softmax_network --train_mode no_finetune --budget_mode 1_5_10_20_50_100
python analysis.py CIFAR100 --analysis_trainer cosine_network --train_mode no_finetune --budget_mode 1_5_10_20_50_100