import json, argparse, os
import numpy as np
from glob import glob
from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import tqdm
import pickle
import torch
import random
import utils
from tabulate import tabulate

SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 20


LEGEND_SIZE = BIGGER_SIZE
TITLE_SIZE= BIGGER_SIZE
LABEL_SIZE = BIGGER_SIZE
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_pretty_name(str, verbose=True):
    if str == 'retrain': return "Retraining"

    if str == 'sequential' : return "Sequential"
    if str == 'independent' : return "Independent"

    if str == 'softmax_network': return "Softmax Network"
    if str == 'cosine_network': return "Cosine Network"
    if str == 'deep_metric':
        if verbose:
            return "Deep Metric Learning"
        else:
            return "DML"
    
    # if str == 'softmax': return "Argmax Probability"
    if str == 'softmax': return "Argmax"
    if str == 'uldr': return "ULDR"
    if str == 'entropy': return "Entropy"
    if str == 'uldr_norm_cosine': return "ULDR (cosine dist.)"
    if str == 'coreset': return "Coreset"
    if str == 'coreset_norm_cosine': return "Coreset (cosine dist.)"
    if str == 'random': return "Random"

def get_valmode_str(str, verbose=True):
    if str == None:
        return "Final Epoch"
    elif str == 'randomized':
        return "Best ValAcc (Random ValSet)"
    elif str == 'balanced':
        return "Best ValAcc (Balanced ValSet)"
        

def get_open_name(str):
    if str == 'softmax': return "Softmax"
    if str == 'openmax': return "Openmax"
    if str == 'entropy': return "Entropy"
    if str == 'nn': return "Nearest Neighbor"
    if str == 'nn_cosine': return "Nearest Neighbor (cosine dist.)"

def get_result_str(key, init_size, budget_dict):
    if key in ['combined', 'same_sample']:
        prefix = "Total Labeled Sample: "
        add_num = init_size
    else:
        prefix = "Total Query Budget: "
        add_num = 0
    
    for b in budget_dict:
        prefix += str(b+add_num) + f":({budget_dict[b]:.4f}) " 
    return prefix

class ActiveAnalysisMachine(object):
    """Store all the configs we want to compare
    """
    def __init__(self,
                 active_analysis_save_dir,
                 budget_list,
                 data_config,
                 download_path,
                 active_save_path,
                 active_save_dir,
                 data,
                 TRAINING_METHODS,
                 ACTIVE_TRAIN_MODES,
                 QUERY_METHODS,
                 ACTIVE_QUERY_SCHEMES,
                 RANDOM_SEEDS,
                #  VAL_MODES,
                 silent_mode=False):
        super().__init__()
        self.active_analysis_save_dir = active_analysis_save_dir
        self.data = data

        self.active_save_dir = active_save_dir
        self.active_save_path = active_save_path
        self.data_download_path = download_path

        self.data_config = data_config
        self.budget_list = budget_list

        self.TRAINING_METHODS = TRAINING_METHODS
        self.ACTIVE_TRAIN_MODES = ACTIVE_TRAIN_MODES
        self.QUERY_METHODS = QUERY_METHODS
        self.ACTIVE_QUERY_SCHEMES = ACTIVE_QUERY_SCHEMES
        self.RANDOM_SEEDS = RANDOM_SEEDS
        # self.VAL_MODES = VAL_MODES

    def gather_results(self, draw_open=True):
        finished_exp_dict = {}
        for active_train_mode in self.ACTIVE_TRAIN_MODES:
            if not active_train_mode in finished_exp_dict:
                finished_exp_dict[active_train_mode] = {}
            for training_method in self.TRAINING_METHODS:
                if not training_method in finished_exp_dict[active_train_mode]:
                    finished_exp_dict[active_train_mode][training_method] = {}
                for query_method in self.QUERY_METHODS:
                    if not query_method in finished_exp_dict[active_train_mode][training_method]:
                        finished_exp_dict[active_train_mode][training_method][query_method] = {}
                    for active_query_scheme in self.ACTIVE_QUERY_SCHEMES:
                        if not active_query_scheme in finished_exp_dict[active_train_mode][training_method][query_method]:
                            finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme] = {}
                        for data_rand_seed in self.RANDOM_SEEDS:
                            if not data_rand_seed in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme]:
                                finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed] = {}
                            # for active_val_mode in self.VAL_MODES:
                            #     if not active_val_mode in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed]:
                            #         finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][active_val_mode] = {}
                            paths_dict = utils.prepare_active_learning_dir(self.budget_list,
                                                                            self.active_save_path,
                                                                            self.data_download_path,
                                                                            self.active_save_dir,
                                                                            self.data,
                                                                            self.data_config,
                                                                            data_rand_seed,
                                                                            training_method,
                                                                            active_train_mode,
                                                                            query_method,
                                                                            active_query_scheme,
                                                                        #    active_val_mode,
                                                                            makedir=False)
                            for b in self.budget_list:
                                if os.path.exists(paths_dict['active_test_results'][b]):
                                    test_result = torch.load(paths_dict['active_test_results'][b],
                                                            map_location=torch.device('cpu'))
                                    res = test_result['acc']
                                    # if not b in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][active_val_mode]:
                                    #     finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][active_val_mode][b] = res
                                    if not b in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed]:
                                        finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][b] = res
        return finished_exp_dict
    
    def gather_query_results(self):
        self.query_result_path = os.path.join(self.active_analysis_save_dir,
                                              self.data,
                                              'query_result_analysis')
        for active_train_mode in self.ACTIVE_TRAIN_MODES:
            for training_method in self.TRAINING_METHODS:
                for query_method in self.QUERY_METHODS:
                    for active_query_scheme in self.ACTIVE_QUERY_SCHEMES:
                        for data_rand_seed in self.RANDOM_SEEDS:
                            paths_dict = utils.prepare_active_learning_dir(self.budget_list,
                                                                            self.active_save_path,
                                                                            self.data_download_path,
                                                                            self.active_save_dir,
                                                                            self.data,
                                                                            self.data_config,
                                                                            data_rand_seed,
                                                                            training_method,
                                                                            active_train_mode,
                                                                            query_method,
                                                                            active_query_scheme,
                                                                        #    active_val_mode,
                                                                            makedir=False)

                            curr_discovered_samples = []
                            for b in self.budget_list:
                                if os.path.exists(paths_dict['active_query_results'][b]):
                                    test_result = torch.load(paths_dict['active_query_results'][b],
                                                             map_location=torch.device('cpu'))
                                    all_result = torch.load(paths_dict['active_test_results'][b],
                                                            map_location=torch.device('cpu'))['closed_set_result']
                                    

                                    trainset_info = torch.load(paths_dict['trainset_info_path'],
                                                            map_location=torch.device('cpu'))
                                    query_samples = np.array(list(set(test_result['new_discovered_samples']).difference(list(curr_discovered_samples))))
                                    curr_discovered_samples = np.array(test_result['new_discovered_samples'])

                                    # Need to plot both query sample and total labeled samples distribution
                                    curr_query_result_path = os.path.join(self.query_result_path,
                                                                          active_train_mode,
                                                                          training_method,
                                                                          "active_"+query_method,
                                                                          active_query_scheme,
                                                                          f'seed_{data_rand_seed}',
                                                                          f'budget_{b}')
                                    if not os.path.exists(curr_query_result_path):
                                        os.makedirs(curr_query_result_path)

                                    for pool_name, samples in [('query_sample', query_samples), ('labeled_pool', curr_discovered_samples)]:
                                        # import pdb; pdb.set_trace()
                                        if len(samples) == 0:
                                            continue
                                        # First calculate the labels and save them in a text file
                                        curr_labels = np.array(trainset_info.train_labels)[samples]
                                        counts_per_class = {}
                                        for l in curr_labels:
                                            if not l in counts_per_class:
                                                counts_per_class[l] = 1
                                            else:
                                                counts_per_class[l] = counts_per_class[l] + 1
                                        with open(os.path.join(curr_query_result_path, pool_name+"_result.txt"), "w+") as file:
                                            result_lists = []
                                            for class_i in sorted(list(counts_per_class.keys())):
                                                result_lists.append([f"{class_i}", f"{counts_per_class[class_i]}"])
                                            file.write(tabulate(result_lists, headers=['Class Index', 'Number of Samples'], tablefmt='orgtbl'))
                                        
                                        # Then plot it as bar charts
                                        plt.figure(figsize=(20,12))
                                        axes = plt.gca()
                                        # axes.set_ylim([-1,1])
                                        plt.title(f'Number of samples in {pool_name} for budget {b}.')
                                        classes = sorted(list(counts_per_class.keys()))
                                        samples_in_classes = [counts_per_class[i] for i in classes]
                                        plt.bar(classes, samples_in_classes, align='center')
                                        # plt.axhline(y=mean_delta, label=f"Mean Accuracy Delta {mean_delta}", linestyle='--', color='black')
                                        classes_tick = [str(i) for i in classes]
                                        plt.xticks(classes, classes_tick)
                                        plt.xlabel('Class label')
                                        plt.ylabel('Number of samples for each class')
                                        plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
                                        # plt.legend()
                                        plt.savefig(os.path.join(curr_query_result_path, pool_name+"_result.png"))
                                        plt.close('all')

                                    
                                    # Calculate the per-class accuracy (recall: pred as cars/all cars)
                                    gt = np.array(all_result['real_labels'])
                                    pred = np.array(all_result['closed_predicted_real'])
                                    overall_acc = float((gt == pred).sum()/pred.shape[0])
                                    all_test_classes_acc = {i : None for i in set(gt) }
                                    for class_i in all_test_classes_acc:
                                        class_i_acc = (gt == pred)[gt == class_i].sum() / (gt == class_i).sum()
                                        all_test_classes_acc[class_i] = class_i_acc
                                    avg_per_class_acc = float(np.array([all_test_classes_acc[i] for i in all_test_classes_acc]).mean())
                                    plt.figure(figsize=(20,12))
                                    axes = plt.gca()
                                    # axes.set_ylim([-1,1])
                                    plt.title(f'Per-class accuracy for budget {b}.')
                                    classes = sorted(list(all_test_classes_acc.keys()))
                                    classes_acc = [all_test_classes_acc[i] for i in all_test_classes_acc]
                                    plt.bar(classes, classes_acc, align='center')
                                    plt.axhline(y=overall_acc, label=f"Overall Test Accuracy {overall_acc}", linestyle='--', color='black')
                                    plt.axhline(y=avg_per_class_acc, label=f"Avg Per-Class Accuracy {avg_per_class_acc}", linestyle='-', color='green')
                                    classes_tick = [str(i) for i in classes]
                                    plt.xticks(classes, classes_tick)
                                    plt.xlabel('Class label')
                                    plt.ylabel('Per-class Accuracy')
                                    plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
                                    plt.legend()
                                    plt.savefig(os.path.join(curr_query_result_path, "per_class_acc.png"))
                                    plt.close('all')

                                    with open(os.path.join(curr_query_result_path, "per_class_acc.txt"), "w+") as file:
                                        result_lists = []
                                        for class_i in classes:
                                            result_lists.append([f"{class_i}", f"{all_test_classes_acc[class_i]}"])
                                        file.write(tabulate(result_lists, headers=['Class Index', 'Per-Class Accuracy/Recall'], tablefmt='orgtbl'))

                                    # Calculate the per-class precision (: pred as car & real car/pred as cars)
                                    all_test_classes_precision = {i : None for i in set(gt) }
                                    for class_i in all_test_classes_precision:
                                        class_i_precision = (gt == class_i)[pred == class_i].sum() / (pred == class_i).sum()
                                        all_test_classes_precision[class_i] = class_i_precision
                                    avg_per_class_precision = float(np.array([all_test_classes_precision[i] for i in all_test_classes_precision]).mean())
                                    plt.figure(figsize=(20,12))
                                    axes = plt.gca()
                                    # axes.set_ylim([-1,1])
                                    plt.title(f'Per-class precision (i.e. (Pred as car && is car)/(Pred as car)) for budget {b}.')
                                    classes = sorted(list(all_test_classes_precision.keys()))
                                    classes_precision = [all_test_classes_precision[i] for i in all_test_classes_precision]
                                    plt.bar(classes, classes_precision, align='center')
                                    plt.axhline(y=avg_per_class_precision, label=f"Avg Per-Class Precision {avg_per_class_precision}", linestyle='-', color='green')
                                    classes_tick = [str(i) for i in classes]
                                    plt.xticks(classes, classes_tick)
                                    plt.xlabel('Class label')
                                    plt.ylabel('Per-class Precision')
                                    plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
                                    plt.legend()
                                    plt.savefig(os.path.join(curr_query_result_path, "per_class_precision.png"))
                                    plt.close('all')

                                    with open(os.path.join(curr_query_result_path, "per_class_precision.txt"), "w+") as file:
                                        result_lists = []
                                        for class_i in classes:
                                            result_lists.append([f"{class_i}", f"{all_test_classes_precision[class_i]}"])
                                        file.write(tabulate(result_lists, headers=['Class Index', 'Per-Class Precision'], tablefmt='orgtbl'))
                                        

    def plot_results(self, finished_exp_dict, plot_mode=None):
        if plot_mode == None:
            # Plot the balanced mode, and the random seed mode with error bars
            
            self.plot_dir = os.path.join(self.active_analysis_save_dir,self.data)
            
            print_setup_dict = {"balanced" : {'data_rand_seed_list' : [None],
                                              'error_bar' : False,
                                              'plot_dir' : os.path.join(self.plot_dir, "balanced_init_set")},
                                "randomized" : {'data_rand_seed_list' : [1, 10, 100, 1000, 2000],
                                                'error_bar' : True,
                                                'plot_dir' : os.path.join(self.plot_dir, "randomized_init_set")},
            }
            for setup in print_setup_dict:
                self._plot_helper(finished_exp_dict, **print_setup_dict[setup])
    
    def _plot_helper(self, finished_exp_dict, data_rand_seed_list=[], error_bar=False, plot_dir=None):
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        for active_train_mode in finished_exp_dict.keys():
            results = {}
            for training_method in finished_exp_dict[active_train_mode].keys():
                for query_method in finished_exp_dict[active_train_mode][training_method].keys():
                    for active_query_scheme in finished_exp_dict[active_train_mode][training_method][query_method].keys():
                        for data_rand_seed in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme].keys():
                            # for active_val_mode in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed].keys():
                            if not data_rand_seed in data_rand_seed_list:
                                continue
                            has_all_budget = True
                            
                            acc = []
                            for b in self.budget_list:
                                # if active_query_scheme == 'independent' and training_method == 'softmax_network' and query_method == 'uldr' and data_rand_seed == 1:
                                #     import pdb; pdb.set_trace()
                                # if not b in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][active_val_mode].keys():
                                if not b in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed].keys():
                                    has_all_budget = False
                                    break
                                # acc.append(float(finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][active_val_mode][b]))
                                acc.append(float(finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][b]))

                            if has_all_budget == True:
                                # if not (training_method, query_method, active_query_scheme, active_val_mode) in results:
                                #     results[(training_method, query_method, active_query_scheme, active_val_mode)] = {}
                                # results[(training_method, query_method, active_query_scheme, active_val_mode)][data_rand_seed] = acc
                                if not (training_method, query_method, active_query_scheme) in results:
                                    results[(training_method, query_method, active_query_scheme)] = {}
                                results[(training_method, query_method, active_query_scheme)][data_rand_seed] = acc
        # Get init set size
        from utils import get_trainset_info_path
        trainset_info = torch.load(get_trainset_info_path(self.active_save_path, self.data))
        if self.data in ['CIFAR100', 'CIFAR10']:
            init_size = DATASET_CONFIG_DICT[self.data][self.data_config]['num_init_classes'] * DATASET_CONFIG_DICT[self.data][self.data_config]['sample_per_class']

        color_dict = {}
        color_list = ['r','b','g', 'c', 'm', 'y', 'black', 'darkblue', 'skyblue', 'steelblue', 'olive', 'deeppink', 'crimson']
        def get_color_func(s):
            if not s in color_dict:
                random.seed(s)
                # print(len(color_list))
                c = random.choice(color_list)
                color_list.remove(c)
                color_dict[s] = c
            return color_dict[s]

        from analysis import TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
        plt.figure(figsize=(15,12))
        plt.title(f'Multi-class classification accuracy plot', fontsize=TITLE_SIZE)
        plt.xlabel("Total number of labeled samples", fontsize=LABEL_SIZE)
        plt.ylabel("Accuracy", fontsize=LABEL_SIZE)

        axes = plt.gca()
        axes.set_ylim([0,1])
        axes.set_xlim([0,init_size + max(self.budget_list)])
        x = [b+init_size for b in self.budget_list]
        for setup in results.keys():
            # training_method, query_method, active_query_scheme, active_val_mode = setup
            # c = get_color_func((query_method,active_query_scheme, active_val_mode))
            training_method, query_method, active_query_scheme = setup
            c = get_color_func((query_method,active_query_scheme))
            if error_bar and len(results[setup].keys()) > 1:
                all_y = np.zeros((len(results[setup].keys()), len(self.budget_list)))
                for i, s in enumerate(results[setup].keys()):
                    all_y[i, :] = np.array(results[setup][s])
                y = all_y.mean(axis=0)
                err = all_y.std(axis=0)
                plt.errorbar(x, y, yerr=err, ls="None", color=c)
            else:
                y = results[setup][list(results[setup].keys())[0]]
            
            # label_str = " | ".join([get_pretty_name(active_query_scheme), "Train = "+get_pretty_name(training_method), "Active = "+get_pretty_name(query_method), "ModelCKPT = "+get_valmode_str(active_val_mode)])
            label_str = " | ".join([get_pretty_name(active_query_scheme), "Train = "+get_pretty_name(training_method), "Active = "+get_pretty_name(query_method)])
            plt.plot(x,
                     y,
                     label=label_str,
                     color=c
                     )

                
        plt.legend(fontsize=LEGEND_SIZE)

        plt.tight_layout()
        save_path = os.path.join(plot_dir, "plot.png")
        print(f"Plot saved at {save_path}")
        plt.savefig(save_path)
        plt.close('all')

    def print_results(self, finished_exp_dict, print_mode=None):
        if print_mode == None:
            if not os.path.exists(os.path.join(self.active_analysis_save_dir,
                                           self.data)):
                os.makedirs(os.path.join(self.active_analysis_save_dir,
                                           self.data,))
            self.print_path = os.path.join(self.active_analysis_save_dir,
                                           self.data,
                                           "results.txt")
            print("Print all results to " + self.print_path)
            with open(self.print_path, "w+") as file:
                for active_train_mode in finished_exp_dict.keys():
                    file.write(f"#####Start Train mode {active_train_mode}######\n")
                    result_lists = []
                    for training_method in finished_exp_dict[active_train_mode].keys():
                        for query_method in finished_exp_dict[active_train_mode][training_method].keys():
                            for active_query_scheme in finished_exp_dict[active_train_mode][training_method][query_method].keys():
                                for data_rand_seed in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme].keys():
                                    # for active_val_mode in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed].keys():
                                    single_list = [get_pretty_name(active_query_scheme),
                                                #    get_valmode_str(active_val_mode),
                                                    str(data_rand_seed),
                                                    get_pretty_name(training_method),
                                                    get_pretty_name(query_method)]
                                    for b in self.budget_list:
                                        # if active_query_scheme == 'independent' and training_method == 'softmax_network' and query_method == 'uldr' and data_rand_seed == 1:
                                        #     import pdb; pdb.set_trace()
                                        if b in finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed].keys():
                                            # single_list += [f"{finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][active_val_mode][b]:.4f}"]
                                            single_list += [f"{finished_exp_dict[active_train_mode][training_method][query_method][active_query_scheme][data_rand_seed][b]:.4f}"]
                                        else:
                                            single_list += ["N/A"]
                                    result_lists.append(single_list)
                    # file.write(tabulate(result_lists, headers=['Query Scheme', 'Model CKPT', 'Seed', 'Training Method', 'Query Method'] + list(map(str, self.budget_list)), tablefmt='orgtbl'))
                    file.write(tabulate(result_lists, headers=['Query Scheme', 'Seed', 'Training Method', 'Query Method'] + list(map(str, self.budget_list)), tablefmt='orgtbl'))
                    file.write(f"\n#####End Train mode {active_train_mode}######\n\n\n")              


if __name__ == "__main__":
    from config import get_config
    from global_setting import DATASET_CONFIG_DICT
    from utils import prepare_save_dir
    config = get_config()

    ACTIVE_QUERY_SCHEMES = ['sequential']
    # ACTIVE_QUERY_SCHEMES = ['independent']
    # ACTIVE_QUERY_SCHEMES = ['sequential']
    ACTIVE_TRAIN_MODES = ['retrain']
    TRAINING_METHODS = ['softmax_network']
    QUERY_METHODS = ['coreset', 'random', 'softmax', 'entropy']
    # QUERY_METHODS = ['coreset', 'random', 'softmax']
    # QUERY_METHODS = ['random']
    # QUERY_METHODS = ['softmax']
    # QUERY_METHODS = ['uldr', 'random']
    # QUERY_METHODS = ['coreset']
    RANDOM_SEEDS = [None, 1, 10, 100, 1000, 2000]

    budget_list = utils.get_budget_list_from_config(config)

    # QUERY_METHODS = ['uldr', 'coreset']
    analysis_machine = ActiveAnalysisMachine(config.active_analysis_save_dir,
                                             budget_list,
                                             config.data_config,
                                             config.data_download_path,
                                             config.active_save_path,
                                             config.active_save_dir,
                                             config.data,
                                             TRAINING_METHODS,
                                             ACTIVE_TRAIN_MODES,
                                             QUERY_METHODS,
                                             ACTIVE_QUERY_SCHEMES,
                                             RANDOM_SEEDS)

    #### 
    # Check all checkpoint files exist
    results = analysis_machine.gather_results()
    analysis_machine.print_results(results, print_mode=None)
    analysis_machine.plot_results(results, plot_mode=None)

    analysis_machine.gather_query_results() # then plot them
    # analysis_machine.draw_closed_set(draw_open=True)
    
    