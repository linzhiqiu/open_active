python analysis.py CIFAR100 --analysis_trainer softmax_network --budget_mode 1_5_10_20_50_100
python analysis.py CIFAR100 --analysis_trainer cosine_network --budget_mode 1_5_10_20_50_100



# No finetune mode
python analysis.py CIFAR100 --analysis_trainer softmax_network --train_mode no_finetune --budget_mode 1_5_10_20_50_100
python analysis.py CIFAR100 --analysis_trainer cosine_network --train_mode no_finetune --budget_mode 1_5_10_20_50_100

# default 200 epoch lr 0.1 finetune
python analysis.py CIFAR100 --analysis_trainer softmax_network --train_mode default_lr01_200eps --budget_mode 1_5_10_20_50_100
python analysis.py CIFAR100 --analysis_trainer cosine_network --train_mode default_lr01_200eps --budget_mode 1_5_10_20_50_100


# default 200 epochs lr 0.1
python analysis.py CIFAR100 --analysis_save_dir /share/coecis/open_active/analysis/paper_result/basic_question/ --analysis_trainer softmax_network --train_mode default_lr01_200eps --budget_mode 1_5_10_20_50_100