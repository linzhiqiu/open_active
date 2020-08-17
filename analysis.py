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
import global_setting
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
    if str == 'default': return "Finetune"
    if str == 'default_lr01_200eps': return "Retrain"
    if str == 'fix_feature_extractor': return "Linear"

    if str == 'regular': return "Original Setup"
    if str == 'fewer_class': return "Fewer Classes"
    if str == 'fewer_sample': return "Fewer Samples"
    
    if str == 'softmax_network': return "Softmax"
    if str == 'cosine_network': return "Cosine"
    if str == 'deep_metric':
        if verbose:
            return "Deep Metric Learning"
        else:
            return "DML"
    
    if str == 'softmax': return "Argmax Probability"
    if str == 'uldr': return "ULDR"
    if str == 'entropy': return "Entropy"
    if str == 'uldr_norm_cosine': return "ULDR (cosine dist.)"
    if str == 'coreset': return "Coreset"
    if str == 'coreset_norm_cosine': return "Coreset (cosine dist.)"
    if str == 'random': return "Random"

def get_open_name(str):
    if str == 'softmax': return "Softmax"
    if str == 'openmax': return "Openmax"
    if str == 'entropy': return "Entropy"
    if str == 'nn': return "Nearest Neighbor"
    if str == 'nn_cosine': return "Nearest Neighbor (cosine dist.)"

def get_label_name(plot_mode, data_config, training_method, query_method, open_method=None, train_mode=None):
    if train_mode == None:
        if plot_mode == 'compare_setting':
            return " w/ ".join(["Network: "+get_pretty_name(training_method), "Query: "+get_pretty_name(query_method)])
        if plot_mode == 'compare_active':
            return " w/ ".join(["Initial Labeled Pool: "+get_pretty_name(data_config), "Query: "+get_pretty_name(query_method)])
        if plot_mode == 'compare_open':
            return "Open: "+get_open_name(open_method)
        else:
            return "_".join([data_config, training_method, query_method])
    else:
        if plot_mode == "compare_active":
            return " w/ ".join(["Update Rule: "+get_pretty_name(train_mode), "Query: "+get_pretty_name(query_method)])
        else:
            return None

def get_result_str(key, init_size, budget_dict):
    if key in ['combined', 'same_sample']:
        prefix = "Total Labeled Sample: "
        add_num = init_size
    else:
        prefix = "Total Query Budget: "
        add_num = 0
    
    for b in budget_dict:
        prefix += str(b+add_num) + f":(mean:{budget_dict[b]['res']:.4f})|(std:{budget_dict[b]['err']:.4f})|(count:{budget_dict[b]['count']:2d}) " 
    return prefix

class AnalysisMachine(object):
    """Store all the configs we want to compare
    """
    def __init__(self, analysis_save_dir, analysis_trainer, budget_mode, data_download_path, dataset_save_path, trainer_save_dir, data, data_rand_seed_list, training_method_list, train_mode, query_method_list):
        super().__init__()
        self.silent_mode = False
        self.analysis_save_dir = analysis_save_dir
        self.analysis_trainer = analysis_trainer
        self.budget_mode = budget_mode
        self.train_mode = train_mode
        self.data = data
        self.save_dir = self.get_save_dir()
        if not os.path.exists(self.save_dir):
            utils.makedirs(self.save_dir)
        else:
            print(f"Already exists: {self.save_dir} . Overwriting")

        self.script_dir = self.get_script_dir()
        self.plot_dir = self.get_plot_dir()

        self.data = data
        
        self.ALL_DATASET_RAND_SEED = data_rand_seed_list
        self.ALL_TRAIN_METHODS = ['softmax_network', 'cosine_network']
        self.ALL_QUERY_METHODS = ['random', 'entropy', 'uldr', 'uldr_norm_cosine', 'coreset', 'coreset_norm_cosine', 'softmax']
        # self.ALL_QUERY_METHODS = ['random', ]
        self.ALL_DATA_CONFIGS = ['regular', 'fewer_class', 'fewer_sample']
        self.ALL_TRAIN_MODES = ['default_lr01_200eps']


        self.PLOT_MODE = ['compare_active', 'compare_train', 'compare_setting']
        self.trainer_save_dir = trainer_save_dir
        self.dataset_save_path = dataset_save_path
        self.data_download_path = data_download_path
        
        self.training_method_list = training_method_list
        self.query_method_list = query_method_list

    def get_save_dir(self):
        return os.path.join(self.analysis_save_dir,
                            self.data,
                            self.train_mode,
                            self.analysis_trainer,
                            self.budget_mode)
    
    def get_script_dir(self):
        return os.path.join(self.get_save_dir(),
                            )
    
    def get_plot_dir(self):
        return os.path.join(self.analysis_save_dir,
                            self.data,
                            self.train_mode,
                            self.budget_mode)
    
    def get_update_plot_dir(self):
        return os.path.join(self.analysis_save_dir,
                            self.data,
                            'all_update_rule',
                            self.budget_mode)
    
    def check_ckpts_exist(self):
        for a_mode in ['same_sample']:
            budget_list_regular, budget_list_fewer= self._get_budget_candidates(analysis_mode=a_mode)
            
            print("For regular setup, the budgets to query are: " + str(budget_list_regular))
            print("For fewer class/sample setup, the budgets to query are: " + str(budget_list_fewer))
            
            print(f"Saving all unfinished experiments to {self.script_dir}")
            undone_exp = []
            script_file = os.path.join(self.script_dir, f"scripts_{a_mode}.sh")
            script_err = os.path.join(self.script_dir, "scripts.err")
            script_out = os.path.join(self.script_dir, "scripts.out")

            enum_list = [('regular', budget_list_regular),
                            ('fewer_class', budget_list_fewer),
                            ('fewer_sample',budget_list_fewer)]
            for data_config, b_list in enum_list:
                print(f"For {data_config} setting: The experiments to run are:")
                undone_exp_mode = []
                for b in b_list:
                    undone_exp_b = []
                    b_dir = os.path.join(self.script_dir, data_config, f"budget_{b}")
                    if not os.path.exists(b_dir): utils.makedirs(b_dir)
                    for training_method in self.training_method_list:
                        for query_method in self.query_method_list:
                            for data_rand_seed in self.ALL_DATASET_RAND_SEED:
                                paths_dict = prepare_save_dir(self.dataset_save_path,
                                                            self.data_download_path,
                                                            self.trainer_save_dir,
                                                            self.data,
                                                            data_config,
                                                            data_rand_seed,
                                                            training_method,
                                                            self.train_mode,
                                                            query_method,
                                                            b,
                                                            global_setting.OPEN_SET_METHOD_DICT[training_method],
                                                            makedir=False)
                                # for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path']:
                                    # for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path', 'open_result_path']:
                                
                                # For all ckpts
                                for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path', 'open_result_paths']:
                                    exp_finished = True
                                    if k == 'open_result_paths':
                                        for o_method in global_setting.OPEN_SET_METHOD_DICT[training_method]:
                                            if not os.path.exists(paths_dict[k][o_method]):
                                                exp_finished = False
                                    else:
                                        if not os.path.exists(paths_dict[k]):
                                            exp_finished = False
                                    if not exp_finished:
                                        python_script = self._get_exp_name(data_config,
                                                                            training_method,
                                                                            query_method,
                                                                            b,
                                                                            data_rand_seed,
                                                                            silent=self.silent_mode)
                                        idx = len(undone_exp_b)
                                        b_err_i = os.path.join(b_dir, f"{idx}.err")
                                        b_out_i = os.path.join(b_dir, f"{idx}.out")
                                        # script = python_script + f" >> >(tee -a {b_out_i} >> {script_out}) 2>> >(tee -a {b_err_i} >> {script_err}) \n"
                                        if self.silent_mode:
                                            script = python_script + f" > {b_out_i} 2> {b_err_i} \n"
                                        else:
                                            script = python_script + " \n"
                                        undone_exp_b.append(script)
                                        break
                    if undone_exp_b.__len__() > 0:
                        print(f"Budget {b}: {len(undone_exp_b)} experiments to run.")
                        undone_exp_mode = undone_exp_mode + undone_exp_b   
                if undone_exp_mode.__len__() > 0:
                    print(f"Mode {data_config}: {len(undone_exp_mode)} to run.")
                    undone_exp = undone_exp + undone_exp_mode
            if undone_exp.__len__() > 0:
                # if os.path.exists(script_file):
                #     input(f"{script_file} already exists. Overwrite >> ")
                if not os.path.exists(b_dir):
                    utils.makedirs(b_dir)
                    print(f"Details will be saved at {script_dir}")
                with open(script_file, "w+") as file:
                    for i, line in enumerate(undone_exp):
                        file.write(line)
            print(f"Budget analysis {a_mode}: {len(undone_exp)} experiments to run at {script_file}.")

    def draw_closed_set(self, draw_open=True, draw_perclass=True):
        budget_list_regular, budget_list_fewer_same_budget = self._get_budget_candidates(analysis_mode='same_budget')
        _, budget_list_fewer_same_sample = self._get_budget_candidates(analysis_mode='same_sample')
        budget_list_fewer = list(set(budget_list_fewer_same_budget + budget_list_fewer_same_sample))
        budget_list_fewer.sort()
        print("For regular setup, the budgets to query are: " + str(budget_list_regular))
        print("For fewer class/sample setup, the budgets to query are: " + str(budget_list_fewer))
        
        finished_exp = {'regular' : {},
                        'fewer_class' : {},
                        'fewer_sample' : {}}
        finished = 0
        unfinished = 0
        has_trainset_info = False
        for data_config, b_list in [
                                  ('regular', budget_list_regular),
                                  ('fewer_class', budget_list_fewer),
                                  ('fewer_sample',budget_list_fewer)]:
            print(f"For {data_config} setting: The experiments completed are:")
            for b in b_list:
                if not b in finished_exp[data_config]:
                    finished_exp[data_config][b] = {}
                for training_method in self.ALL_TRAIN_METHODS:
                    if not training_method in finished_exp[data_config][b]:
                        finished_exp[data_config][b][training_method] = {}
                    for query_method in self.ALL_QUERY_METHODS:
                        if not query_method in finished_exp[data_config][b][training_method]:
                            finished_exp[data_config][b][training_method][query_method] = {}
                        for data_rand_seed in self.ALL_DATASET_RAND_SEED:
                            paths_dict = prepare_save_dir(self.dataset_save_path,
                                                          self.data_download_path,
                                                          self.trainer_save_dir,
                                                          self.data,
                                                          data_config,
                                                          data_rand_seed,
                                                          training_method,
                                                          self.train_mode,
                                                          query_method,
                                                          b,
                                                          global_setting.OPEN_SET_METHOD_DICT[training_method],
                                                          makedir=False)
                            print(str(b) + " " + training_method + " " + query_method)
                            if os.path.exists(paths_dict['test_result_path']):
                                test_result = torch.load(paths_dict['test_result_path'], map_location=torch.device('cpu'))
                                finished_exp[data_config][b][training_method][query_method][data_rand_seed] = test_result

                                if draw_perclass:
                                    query_result = torch.load(paths_dict['query_result_path'], map_location=torch.device('cpu'))
                                    finished_exp[data_config][b][training_method][query_method][data_rand_seed]['query_result'] = query_result
                                    dataset_info = torch.load(paths_dict['data_save_path'], map_location=torch.device('cpu'))
                                    finished_exp[data_config][b][training_method][query_method][data_rand_seed]['initial_discovered_classes'] = dataset_info['discovered_classes']
                                    finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_classes'] = dataset_info['open_classes']
                                
                                    if not has_trainset_info:
                                        train_labels = torch.load(paths_dict['trainset_info_path'], map_location=torch.device('cpu')).train_labels
                                        has_trainset_info = True
                                    finished_exp[data_config][b][training_method][query_method][data_rand_seed]['train_labels'] = train_labels
                                finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_results'] = {}
                                
                                for o_method in global_setting.OPEN_SET_METHOD_DICT[training_method]:
                                    if os.path.exists(paths_dict['open_result_paths'][o_method]):
                                        open_result = torch.load(paths_dict['open_result_paths'][o_method], map_location=torch.device('cpu'))
                                        finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_results'][o_method] = {}
                                        finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_results'][o_method]['auroc'] = open_result['roc']['auc_score']
                                        finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_results'][o_method]['roc'] = open_result['roc']
                                        finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_results'][o_method]['augoscr'] = open_result['goscr']['auc_score']
                                        finished_exp[data_config][b][training_method][query_method][data_rand_seed]['open_results'][o_method]['goscr'] = open_result['goscr']
                                finished += 1
                            else:
                                unfinished += 1

        # total = finished+unfinished
        # print(f"{finished}/{total} experiments are finished.")
        # print(f"Plot will be draw at {self.plot_dir}")

        comparsion_dict = {
            'same_sample' : {'path':os.path.join(self.plot_dir, "same_sample"),
                             'fewer_b_list': budget_list_fewer_same_sample,
                             'regular_b_list': budget_list_regular},
            # 'same_budget' : {'path':os.path.join(self.plot_dir, "same_budget"),
            #                  'fewer_b_list': budget_list_fewer_same_budget,
            #                  'regular_b_list': budget_list_regular},
            # 'combined' : {'path':os.path.join(self.plot_dir, "combined"),
            #               'fewer_b_list': budget_list_fewer,
            #               'regular_b_list': budget_list_regular},
        }
        if not os.path.exists(self.plot_dir): utils.makedirs(self.plot_dir)
        print("All plots are saved at " + self.plot_dir)
        total_pool_size, regular_init_size, fewer_init_size, budget_ratio = self._get_dataset_info()
        
        
        for k in comparsion_dict.keys():
            if draw_perclass: self._draw_perclass_plot(finished_exp, k, comparsion_dict, total_pool_size, regular_init_size, fewer_init_size, budget_ratio)
            for plot_mode in self.PLOT_MODE:
                self._draw_closed_set_plot(plot_mode, finished_exp, k, comparsion_dict, total_pool_size, regular_init_size, fewer_init_size, budget_ratio)
            if draw_open: self._draw_open_set_plot(finished_exp, k, comparsion_dict, total_pool_size, regular_init_size, fewer_init_size, budget_ratio)

    def _draw_closed_set_plot(self, plot_mode, finished_exp, key, comparsion_dict, total_size, regular_size, fewer_size, budget_ratio, draw_seen_line=True, draw_acc_lowest=True, draw_acc_highest=True, error_bar=False):
        assert key in comparsion_dict
        path = os.path.join(comparsion_dict[key]['path'], plot_mode)
        if not os.path.exists(path): utils.makedirs(path)
        fewer_b_list = comparsion_dict[key]['fewer_b_list']
        regular_b_list = comparsion_dict[key]['regular_b_list']
        
        min_seen_line_idx_dict = {}
        for item in ['seen', 'acc']:
            if plot_mode == 'compare_active':
                COMPARARISON = self.ALL_QUERY_METHODS
            elif plot_mode == 'compare_train':
                COMPARARISON = self.ALL_TRAIN_METHODS
            elif plot_mode == 'compare_setting':
                COMPARARISON = self.ALL_DATA_CONFIGS
        
            for compare_thing in COMPARARISON:
                if not compare_thing in min_seen_line_idx_dict:
                    min_seen_line_idx_dict[compare_thing] = 0
                acc_min = 1.
                acc_max = 0.
                plt.figure(figsize=(15,12))
                if item == 'acc': plt.title(f'Closed set accuracy', fontsize=TITLE_SIZE); plt.ylabel(f"Accuracy", fontsize=LABEL_SIZE)
                if item == 'seen': plt.title(f'Class discovered rate', fontsize=TITLE_SIZE); plt.ylabel(f"Discovered Rate", fontsize=LABEL_SIZE)
                
                lines = 0
                color_dict = {}
                color_list = ['r','b','g', 'c', 'm', 'y', 'black', 'darkblue']
                marker_dict = {}
                marker_list = [',', '+', '.', 'o', '*', 'p', 'D']
                def get_color_func(s):
                    if not s in color_dict:
                        random.seed(s)
                        # print(len(color_list))
                        c = random.choice(color_list)
                        color_list.remove(c)
                        color_dict[s] = c
                    return color_dict[s]

                def get_marker_func(s):
                    if not s in marker_dict:
                        # print(s)
                        random.seed(s)
                        c = random.choice(marker_list)
                        marker_list.remove(c)
                        marker_dict[s] = c
                    return marker_dict[s]

                def get_style_funcs(plot_mode):
                    if plot_mode == 'compare_active':
                        color_func = lambda i, t, a: get_color_func(i)
                        marker_func = lambda i, t, a: get_marker_func(t)
                    elif plot_mode == 'compare_train':
                        color_func = lambda i, t, a: get_color_func(i)
                        marker_func = lambda i, t, a: get_marker_func(a)
                    elif plot_mode == 'compare_setting':
                        color_func = lambda i, t, a: get_color_func(a)
                        marker_func = lambda i, t, a: get_marker_func(t)
                    return color_func, marker_func

                color_func, marker_func = get_style_funcs(plot_mode)
                axes = plt.gca()
                axes.set_xlim([0, total_size])
                if key == 'same_sample':
                    plt.xlabel("Number of total labeled samples", fontsize=LABEL_SIZE)
                # elif key == 'same_budget':
                #     plt.xlabel("Number of budgets after initial round", fontsize=LABEL_SIZE)
                # elif key == 'combined':
                #     plt.xlabel("Number of total labeled samples", fontsize=LABEL_SIZE)
                
                save_path = os.path.join(path, compare_thing+"_"+item+".png")
                save_path_txt = os.path.join(path, compare_thing+"_"+item+".txt")
                detail_dict = {}

                ALL_DATA_CONFIGS = self.ALL_DATA_CONFIGS
                ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
                ALL_QUERY_METHODS = self.ALL_QUERY_METHODS
                ALL_DATASET_RAND_SEED = self.ALL_DATASET_RAND_SEED
                if plot_mode == 'compare_active':
                    ALL_QUERY_METHODS = [compare_thing]
                elif plot_mode == 'compare_train':
                    ALL_TRAIN_METHODS = [compare_thing]
                else:
                    ALL_DATA_CONFIGS = [compare_thing]

                for data_config in ALL_DATA_CONFIGS:
                    if not data_config in detail_dict: detail_dict[data_config] = {}

                    if data_config in 'regular':
                        budget_list = regular_b_list
                        init_size = regular_size
                    else:
                        budget_list = fewer_b_list
                        init_size = fewer_size
                    x = np.array(budget_list)
                    if key in ['combined', 'same_sample']:
                        x = x + init_size
                    for training_method in ALL_TRAIN_METHODS:
                        if not training_method in detail_dict: detail_dict[data_config][training_method] = {}
                        for query_method in ALL_QUERY_METHODS:
                            if not query_method in detail_dict: detail_dict[data_config][training_method][query_method] = {}
                            y = np.array([None for _ in x]).astype(np.double)
                            err = np.array([None for _ in x]).astype(np.double)
                            for idx, b in enumerate(budget_list):
                                is_ready = False
                                if b in finished_exp[data_config]:
                                    if training_method in finished_exp[data_config][b]:
                                        if query_method in finished_exp[data_config][b][training_method]:
                                            if len(finished_exp[data_config][b][training_method][query_method].keys()) >= 1:
                                                # at least one experiment is finished
                                                is_ready = True
                                if is_ready:
                                    all_res = np.array([float(finished_exp[data_config][b][training_method][query_method][s][item]) for s in finished_exp[data_config][b][training_method][query_method].keys()])
                                    res = float(all_res.mean())
                                    std = float(all_res.std())
                                    detail_dict[data_config][training_method][query_method][b] = {'res': res, 'err' : std, 'count' : len(finished_exp[data_config][b][training_method][query_method].keys())}
                                    y[idx] = res
                                    err[idx] = std
                                    if draw_seen_line and item=='seen' and res < 1 and min_seen_line_idx_dict[compare_thing] <= idx:
                                        # if key == 'same_budget' and data_config == 'fewer_sample':
                                        #     import pdb; pdb.set_trace()
                                        min_seen_line_idx_dict[compare_thing] = idx+1
                                    if draw_acc_lowest and item=='acc' and res < acc_min:
                                        acc_min = res
                                    if draw_acc_highest and item=='acc' and res > acc_max:
                                        acc_max = res
                            if np.any(np.isfinite(y)):
                                lines += 1
                                if error_bar:
                                    plt.errorbar(x[np.isfinite(y)], y[np.isfinite(y)], yerr=err[np.isfinite(y)], ls="None", color=c)
                                label_str = get_label_name(plot_mode, data_config, training_method, query_method, open_method=None)
                                c = color_func(data_config, training_method, query_method)
                                m = marker_func(data_config, training_method, query_method)
                                plt.plot(x[np.isfinite(y)],
                                         y[np.isfinite(y)],
                                         label=label_str,
                                         color=c,
                                         marker=m)
                plt.legend(fontsize=LEGEND_SIZE)
                
                if draw_seen_line:
                    min_seen = budget_list[min_seen_line_idx_dict[compare_thing]]
                    if key in ['combined', 'same_sample']:
                        min_seen = min_seen + init_size
                    print("all seen line draw at " + str(min_seen))
                    plt.vlines(min_seen,0,1,
                               linestyles='dashed')
                    # plt.plot((min_seen_line,min_seen_line),(0,1))
                y_min = max(0, acc_min - 0.05) if draw_acc_lowest and item=='acc' else 0. 
                y_max = min(1, acc_max + 0.05) if draw_acc_highest and item=='acc' else 1.
                axes.set_ylim([y_min, y_max])
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close('all')

                # Save the detail dict at save_path_txt
                with open(save_path_txt, "w+") as file:
                    result_lists = []
                    for i_mode in detail_dict:
                        for t_method in detail_dict[i_mode]:
                            for q_method in detail_dict[i_mode][t_method]:
                                label_str = get_label_name(plot_mode, i_mode, t_method, q_method, open_method=None)
                                result_str = get_result_str(key, init_size, detail_dict[i_mode][t_method][q_method])
                                result_lists.append([label_str, result_str])
                    file.write(tabulate(result_lists, headers=['Details', 'Result'], tablefmt='orgtbl'))
                print(f"Saved at {save_path_txt}")

        for item in ['auroc', 'augoscr']:
            for o_method in ALL_OPEN_METHODS:
                open_path = os.path.join(path, o_method)
                if not os.path.exists(open_path): utils.makedirs(open_path)
                if plot_mode == 'compare_active':
                    COMPARARISON = self.ALL_QUERY_METHODS
                elif plot_mode == 'compare_train':
                    COMPARARISON = self.ALL_TRAIN_METHODS
                elif plot_mode == 'compare_setting':
                    COMPARARISON = self.ALL_DATA_CONFIGS

                for compare_thing in COMPARARISON:
                    plt.figure(figsize=(15,12))
                    if item == 'auroc': plt.title(f'Area under ROC', fontsize=TITLE_SIZE); plt.ylabel(f"Area under curve", fontsize=LABEL_SIZE)
                    if item == 'augoscr': plt.title(f'Area under OSCR', fontsize=TITLE_SIZE); plt.ylabel(f"Area under curve", fontsize=LABEL_SIZE)
                    lines = 0
                    color_dict = {}
                    color_list = ['r','b','g', 'c', 'm', 'y', 'black', 'darkblue']
                    marker_dict = {}
                    marker_list = [',', '+', '.', 'o', '*', 'p', 'D']
                    style_dict = {}
                    style_list = [',', 'solid', 'dashed', 'dashdot', 'dotted']
                    def get_style_func(s):
                        if not s in style_dict:
                            random.seed(s)
                            c = random.choice(style_list)
                            style_list.remove(c)
                            style_dict[s] = c
                        return style_dict[s]
                    
                    def get_color_func(s):
                        if not s in color_dict:
                            random.seed(s)
                            # print(len(color_list))
                            c = random.choice(color_list)
                            color_list.remove(c)
                            color_dict[s] = c
                        return color_dict[s]

                    def get_marker_func(s):
                        if not s in marker_dict:
                            # print(s)
                            random.seed(s)
                            c = random.choice(marker_list)
                            marker_list.remove(c)
                            marker_dict[s] = c
                        return marker_dict[s]

                    def get_style_funcs(plot_mode):
                        if plot_mode == 'compare_active':
                            color_func = lambda i, t, a: get_color_func(i)
                            marker_func = lambda i, t, a: get_marker_func(t)
                        elif plot_mode == 'compare_train':
                            color_func = lambda i, t, a: get_color_func(i)
                            marker_func = lambda i, t, a: get_marker_func(a)
                        elif plot_mode == 'compare_setting':
                            color_func = lambda i, t, a: get_color_func(a)
                            marker_func = lambda i, t, a: get_marker_func(t)
                        return color_func, marker_func

                    color_func, marker_func = get_style_funcs(plot_mode)
                    axes = plt.gca()
                    axes.set_ylim([0,1])
                    axes.set_xlim([0, total_size])
                    if key == 'same_sample':
                        plt.xlabel("Number of total labeled samples", fontsize=LABEL_SIZE)
                    elif key == 'same_budget':
                        plt.xlabel("Number of budgets after initial round", fontsize=LABEL_SIZE)
                    elif key == 'combined':
                        plt.xlabel("Number of total labeled samples", fontsize=LABEL_SIZE)
                    
                    save_path = os.path.join(open_path, compare_thing+"_"+item+".png")
                    save_path_txt = os.path.join(open_path, compare_thing+"_"+item+".txt")
                    detail_dict = {}

                    ALL_DATA_CONFIGS = self.ALL_DATA_CONFIGS
                    ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
                    ALL_QUERY_METHODS = self.ALL_QUERY_METHODS
                    if plot_mode == 'compare_active':
                        ALL_QUERY_METHODS = [compare_thing]
                    elif plot_mode == 'compare_train':
                        ALL_TRAIN_METHODS = [compare_thing]
                    else:
                        ALL_DATA_CONFIGS = [compare_thing]

                    for data_config in ALL_DATA_CONFIGS:
                        if not data_config in detail_dict: detail_dict[data_config] = {}
                        if data_config in 'regular':
                            budget_list = regular_b_list
                            init_size = regular_size
                        else:
                            budget_list = fewer_b_list
                            init_size = fewer_size
                        x = np.array(budget_list)
                        if key in ['combined', 'same_sample']:
                            x = x + init_size
                        for training_method in ALL_TRAIN_METHODS:
                            if not training_method in detail_dict: detail_dict[data_config][training_method] = {}
                            for query_method in ALL_QUERY_METHODS:
                                if not query_method in detail_dict: detail_dict[data_config][training_method][query_method] = {}
                                y = np.array([None for _ in x]).astype(np.double)
                                err = np.array([None for _ in x]).astype(np.double)
                                for idx, b in enumerate(budget_list):
                                    is_ready = False
                                    count = 0
                                    if b in finished_exp[data_config]:
                                        if training_method in finished_exp[data_config][b]:
                                            if query_method in finished_exp[data_config][b][training_method]:
                                                for s in finished_exp[data_config][b][training_method][query_method].keys():
                                                    if o_method in finished_exp[data_config][b][training_method][query_method][s]['open_results']:
                                                        is_ready = True
                                                        count += 1
                                    if is_ready:
                                        all_res = np.array([float(finished_exp[data_config][b][training_method][query_method][s]['open_results'][o_method][item]) for s in finished_exp[data_config][b][training_method][query_method].keys()])
                                        # res = float(finished_exp[data_config][b][training_method][query_method]['open_results'][o_method][item])
                                        res = float(all_res.mean())
                                        std = float(all_res.std())
                                        detail_dict[data_config][training_method][query_method][b] = {'res': res, 'err' : std, 'count' : count}
                                        y[idx] = res
                                if np.any(np.isfinite(y)):
                                    lines += 1
                                    label_str = get_label_name(plot_mode, data_config, training_method, query_method, open_method=o_method)
                                    if error_bar:
                                        plt.errorbar(x[np.isfinite(y)], y[np.isfinite(y)], yerr=err[np.isfinite(y)], ls="None", color=c)
                                    c = color_func(data_config, training_method, query_method)
                                    m = marker_func(data_config, training_method, query_method)
                                    plt.plot(x[np.isfinite(y)],
                                            y[np.isfinite(y)],
                                            label=label_str,
                                            color=c,
                                            # linestyle=get_style_func(o_getmethod),
                                            marker=m)
                    # plt.legend(fontsize=LEGEND_SIZE)
                    if draw_seen_line:
                        min_seen = budget_list[min_seen_line_idx_dict[compare_thing]]
                        if key in ['combined', 'same_sample']:
                            min_seen = min_seen + init_size
                        # print("all seen line draw at " + str(min_seen))
                        plt.vlines(min_seen,0,1,
                                   linestyles='dashed')
                    plt.tight_layout()
                    # print(save_path + f"has {lines} lines.")
                    plt.savefig(save_path)
                    plt.close('all')
                    # Save the detail dict at save_path_txt
                    with open(save_path_txt, "w+") as file:
                        result_lists = []
                        for i_mode in detail_dict:
                            for t_method in detail_dict[i_mode]:
                                for q_method in detail_dict[i_mode][t_method]:
                                    label_str = get_label_name(plot_mode, i_mode, t_method, q_method, open_method=None)
                                    init_size = regular_size if i_mode == "regular" else fewer_size
                                    result_str = get_result_str(key, init_size, detail_dict[i_mode][t_method][q_method])
                                    result_lists.append([label_str, result_str])
                        file.write(tabulate(result_lists, headers=['Details', 'Result'], tablefmt='orgtbl'))
                    print(f"Saved at {save_path_txt}")
    
    def _draw_open_set_plot(self, finished_exp, key, comparsion_dict, total_size, regular_size, fewer_size, budget_ratio, draw_seen_line=True, error_bar=True):
        assert key in comparsion_dict
        path = os.path.join(comparsion_dict[key]['path'], "compare_open")
        if not os.path.exists(path): utils.makedirs(path)
        fewer_b_list = comparsion_dict[key]['fewer_b_list']
        regular_b_list = comparsion_dict[key]['regular_b_list']
        
        ALL_DATA_CONFIGS = self.ALL_DATA_CONFIGS
        ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
        ALL_QUERY_METHODS = self.ALL_QUERY_METHODS

        if draw_seen_line:
            min_seen_line_idx_dict = {}
            for data_config in ALL_DATA_CONFIGS:
                if not data_config in min_seen_line_idx_dict: min_seen_line_idx_dict[data_config] = {}
                for training_method in ALL_TRAIN_METHODS:
                    if not training_method in min_seen_line_idx_dict[data_config]: min_seen_line_idx_dict[data_config][training_method] = {}
                    for query_method in ALL_QUERY_METHODS:
                        for b in sorted(finished_exp[data_config]):
                            if training_method in finished_exp[data_config][b]:
                                if query_method in finished_exp[data_config][b][training_method]:
                                    seen_rates = np.array([float(finished_exp[data_config][b][training_method][query_method][s]['seen']) for s in finished_exp[data_config][b][training_method][query_method]])
                                    if seen_rates.mean() == 1:
                                        if not query_method in min_seen_line_idx_dict[data_config][training_method]:
                                            min_seen_line_idx_dict[data_config][training_method][query_method] = b
        
        for item in ['auroc', 'augoscr']:
            for data_config in ALL_DATA_CONFIGS:
                if data_config in 'regular':
                    budget_list = regular_b_list
                    init_size = regular_size
                else:
                    budget_list = fewer_b_list
                    init_size = fewer_size
                x = np.array(budget_list)
                if key in ['combined', 'same_sample']:
                    x = x + init_size
                for training_method in ALL_TRAIN_METHODS:
                    for query_method in ALL_QUERY_METHODS:
                        color_dict = {}
                        color_list = ['r','b','g', 'c', 'm', 'y', 'black', 'darkblue']
                        def get_color_func(s):
                            if not s in color_dict:
                                random.seed(s)
                                # print(len(color_list))
                                c = random.choice(color_list)
                                color_list.remove(c)
                                color_dict[s] = c
                            return color_dict[s]
                        detail_dict = {}
                        plt.figure(figsize=(15,12))
                        if item == 'auroc': plt.title(f'Area under ROC', fontsize=TITLE_SIZE); plt.ylabel(f"Area under curve", fontsize=LABEL_SIZE)
                        if item == 'augoscr': plt.title(f'Area under GOSCR', fontsize=TITLE_SIZE); plt.ylabel(f"Area under curve", fontsize=LABEL_SIZE)
                        axes = plt.gca()
                        axes.set_ylim([0,1])
                        axes.set_xlim([0, total_size])
                        if key == 'same_sample':
                            plt.xlabel("Number of total labeled samples", fontsize=LABEL_SIZE)
                        elif key == 'same_budget':
                            plt.xlabel("Number of budgets after initial round", fontsize=LABEL_SIZE)
                        elif key == 'combined':
                            plt.xlabel("Number of total labeled samples", fontsize=LABEL_SIZE)
                        for o_method in global_setting.OPEN_SET_METHOD_DICT[training_method]:
                            y = np.array([None for _ in x]).astype(np.double)
                            err = np.array([None for _ in x]).astype(np.double)
                            for idx, b in enumerate(budget_list):
                                is_ready = False
                                count = 0
                                if b in finished_exp[data_config]:
                                    if training_method in finished_exp[data_config][b]:
                                        if query_method in finished_exp[data_config][b][training_method]:
                                            for s in finished_exp[data_config][b][training_method][query_method].keys():
                                                if o_method in finished_exp[data_config][b][training_method][query_method][s]['open_results']:
                                                    is_ready = True
                                                    count += 1
                                if is_ready:
                                    if not training_method in detail_dict: detail_dict[training_method] = {}
                                    if not query_method in detail_dict[training_method]: detail_dict[training_method][query_method] = {}
                                    if not o_method in detail_dict[training_method][query_method]: detail_dict[training_method][query_method][o_method] = {}
                                    all_res = np.array([float(finished_exp[data_config][b][training_method][query_method][s]['open_results'][o_method][item]) for s in finished_exp[data_config][b][training_method][query_method]])
                                    res = float(all_res.mean())
                                    std = float(all_res.std())
                                    detail_dict[training_method][query_method][o_method][b] = {'res' : res, 'err': std, 'count' : count}
                                    y[idx] = res
                                    err[idx] = std
                            if np.any(np.isfinite(y)):
                                label_str = get_label_name("compare_open", None, None, None, open_method=o_method)
                                c = get_color_func(o_method)
                                plt.plot(x[np.isfinite(y)],
                                         y[np.isfinite(y)],
                                         label=label_str,
                                         color=c,
                                         marker='.')
                                if error_bar:
                                    plt.errorbar(x[np.isfinite(y)], y[np.isfinite(y)], yerr=err[np.isfinite(y)], ls="None", color=c)
                        plt.legend(fontsize=LEGEND_SIZE)
                        if draw_seen_line:
                            draw_line = False
                            try:
                                # if data_config == 'fewer_sample' and training_method == 'softmax_network':
                                #     import pdb; pdb.set_trace()
                                min_seen = min_seen_line_idx_dict[data_config][training_method][query_method]
                                draw_line = True
                            except:
                                pass
                            if draw_line:
                                if key in ['combined', 'same_sample']:
                                    init_size = fewer_size if data_config != 'regular' else regular_size
                                    min_seen = min_seen + init_size
                                    plt.vlines(min_seen,0,1,
                                            linestyles='dashed')
                            else:
                                print(f"Not drawing line for {data_config}/{training_method}/{query_method}")

                        plt.tight_layout()
                        save_dir = os.path.join(path, data_config, training_method, query_method)
                        utils.makedirs(save_dir)
                        save_path = os.path.join(save_dir, item+".png")
                        save_path_txt = os.path.join(save_dir, item+".txt")
                        plt.savefig(save_path)
                        plt.close('all')

                        # Save the detail dict at save_path_txt
                        with open(save_path_txt, "w+") as file:
                            result_lists = []
                            for t_method in detail_dict:
                                for q_method in detail_dict[t_method]:
                                    for o_method in detail_dict[t_method][q_method]:
                                        label_str = get_label_name("compare_open", data_config, t_method, q_method, open_method=o_method)
                                        result_str = get_result_str(key, init_size, detail_dict[t_method][q_method][o_method])
                                        result_lists.append([label_str, result_str])
                            file.write(tabulate(result_lists, headers=['Details', 'Result'], tablefmt='orgtbl'))
        
        for item in ['roc', 'goscr']:
            for data_config in self.ALL_DATA_CONFIGS:
                if data_config in 'regular':
                    budget_list = regular_b_list
                    init_size = regular_size
                else:
                    budget_list = fewer_b_list
                    init_size = fewer_size
                x = np.array(budget_list)
                if key in ['combined', 'same_sample']:
                    x = x + init_size
                for training_method in self.ALL_TRAIN_METHODS:
                    for query_method in self.ALL_QUERY_METHODS:
                        for seed in self.ALL_DATASET_RAND_SEED:
                            y = np.array([None for _ in x]).astype(np.double)
                            for idx, b in enumerate(budget_list):
                                color_dict = {}
                                color_list = ['r','b','g', 'c', 'm', 'y', 'black', 'darkblue']
                                def get_color_func(s):
                                    if not s in color_dict:
                                        random.seed(s)
                                        # print(len(color_list))
                                        c = random.choice(color_list)
                                        color_list.remove(c)
                                        color_dict[s] = c
                                    return color_dict[s]

                                plt.figure(figsize=(15,12))
                                if item == 'roc': plt.title(f'ROC plot', fontsize=TITLE_SIZE); plt.xlabel("False Positive Rate (Closed set examples classified as open set)", fontsize=LABEL_SIZE); plt.ylabel("True Positive Rate (Open set example classified as open set)", fontsize=LABEL_SIZE)
                                if item == 'goscr': plt.title(f'OSCR plot', fontsize=TITLE_SIZE); plt.xlabel("False Positive Rate (Open set examples classified as closed set)", fontsize=LABEL_SIZE); plt.ylabel("Correct Classification Rate (Closed set examples classified into correct class)", fontsize=LABEL_SIZE)

                                axes = plt.gca()
                                axes.set_ylim([0,1])
                                axes.set_xlim([0, 1])

                                for o_method in global_setting.OPEN_SET_METHOD_DICT[training_method]:
                                    is_ready = False
                                    if b in finished_exp[data_config]:
                                        if training_method in finished_exp[data_config][b]:
                                            if query_method in finished_exp[data_config][b][training_method]:
                                                if seed in finished_exp[data_config][b][training_method][query_method]:
                                                    if o_method in finished_exp[data_config][b][training_method][query_method][seed]['open_results']:
                                                        is_ready = True
                                    if is_ready:
                                        res = finished_exp[data_config][b][training_method][query_method][seed]['open_results'][o_method][item]
                                        x = res['fpr']
                                        if item == 'roc':
                                            y = res['tpr']
                                            plt.title(f'Receiver Operating Characteristic plot', y=0.96, fontsize=TITLE_SIZE)
                                        else:
                                            y = res['tcr']
                                            axes.set_xscale('log')
                                            axes.autoscale(enable=True, axis='x', tight=True)
                                            plt.title(f'Open set classification rate plot', y=0.96, fontsize=TITLE_SIZE)
                                    if np.any(np.isfinite(y)):
                                        auc_score = res['auc_score']
                                        label_str = " w/ ".join([get_open_name(o_method), "AUC score = "+f"{auc_score:.4f}"])
                                        c = get_color_func(o_method)
                                        plt.plot(x[np.isfinite(y)],
                                                y[np.isfinite(y)],
                                                label=label_str,
                                                color=c)
                                if item == 'roc':
                                    plt.legend(loc='lower right',
                                            borderaxespad=0., fontsize=LEGEND_SIZE)
                                else:
                                    plt.legend(loc='upper left',
                                            borderaxespad=0., fontsize=LEGEND_SIZE)

                                plt.tight_layout()
                                save_dir = os.path.join(path, data_config, training_method, query_method, f"budget_{b}", f"seed_{seed}")
                                utils.makedirs(save_dir)
                                save_path = os.path.join(save_dir, item+".png")
                                plt.savefig(save_path)
                                plt.close('all')

    def _draw_perclass_plot(self, finished_exp, key, comparsion_dict, total_size, regular_size, fewer_size, budget_ratio, draw_seen_line=True, error_bar=True):
        assert key in comparsion_dict
        path = os.path.join(comparsion_dict[key]['path'], "compare_perclass")
        if not os.path.exists(path): utils.makedirs(path)
        fewer_b_list = comparsion_dict[key]['fewer_b_list']
        regular_b_list = comparsion_dict[key]['regular_b_list']
        
        for data_config in self.ALL_DATA_CONFIGS:
            if data_config in 'regular':
                budget_list = regular_b_list
                init_size = regular_size
            else:
                budget_list = fewer_b_list
                init_size = fewer_size
            x = np.array(budget_list)
            if key in ['combined', 'same_sample']:
                x = x + init_size
            for training_method in self.ALL_TRAIN_METHODS:
                for query_method in self.ALL_QUERY_METHODS:
                    for seed in self.ALL_DATASET_RAND_SEED:
                        for idx, b in enumerate(budget_list):
                            is_ready = False
                            if b in finished_exp[data_config]:
                                if training_method in finished_exp[data_config][b]:
                                    if query_method in finished_exp[data_config][b][training_method]:
                                        if seed in finished_exp[data_config][b][training_method][query_method]:
                                            # if 'softmax' in finished_exp[data_config][b][training_method][query_method][seed]['open_results']:
                                            is_ready = True
                            if is_ready:
                                test_result = finished_exp[data_config][b][training_method][query_method][seed]['query_result']
                                all_result = finished_exp[data_config][b][training_method][query_method][seed]['closed_set_result']
                                initial_discovered_classes = finished_exp[data_config][b][training_method][query_method][seed]['initial_discovered_classes']
                                open_classes = finished_exp[data_config][b][training_method][query_method][seed]['open_classes']
                                train_labels = finished_exp[data_config][b][training_method][query_method][seed]['train_labels']
                                curr_discovered_samples = np.array(test_result['new_discovered_samples'])

                                # Need to plot both query sample and total labeled samples distribution
                                curr_query_result_path = os.path.join(path,
                                                                      data_config,
                                                                      training_method,
                                                                      "active_"+query_method,
                                                                      f'seed_{seed}',
                                                                      f'budget_{b}')
                                if not os.path.exists(curr_query_result_path):
                                    os.makedirs(curr_query_result_path)

                                pool_name = 'labeled_pool'
                                samples = curr_discovered_samples
                                # First calculate the labels and save them in a text file
                                curr_labels = np.array(train_labels)[samples]
                                counts_per_class = {}
                                for l in curr_labels:
                                    if not l in counts_per_class:
                                        counts_per_class[l] = 1
                                    else:
                                        counts_per_class[l] = counts_per_class[l] + 1
                                with open(os.path.join(curr_query_result_path, pool_name+"_result.txt"), "w+") as file:
                                    result_lists = []
                                    for class_i in sorted(list(counts_per_class.keys())):
                                        if class_i not in open_classes:
                                            if class_i in initial_discovered_classes:
                                                class_str = str(class_i) + " (initial labeled set)"
                                            else:
                                                class_str = str(class_i) + " (newly discovered)"
                                        else:
                                            class_str = str(class_i) + " (open)"
                                        result_lists.append([f"{class_str}", f"{counts_per_class[class_i]}"])
                                    file.write(tabulate(result_lists, headers=['Class Index', 'Number of Samples'], tablefmt='orgtbl'))
                                
                                # import pdb; pdb.set_trace()
                                # Then plot it as bar charts
                                plt.figure(figsize=(20,12))
                                axes = plt.gca()
                                # axes.set_ylim([-1,1])
                                plt.title(f'Number of samples in {pool_name} for budget {b}.')
                                classes = sorted(list(counts_per_class.keys()))
                                samples_in_classes = [counts_per_class[i] for i in classes]
                                classes_color = ['red' if i in initial_discovered_classes else 'green' for i in classes]
                                plt.bar(classes, samples_in_classes, align='center', color=classes_color)
                                # plt.axhline(y=mean_delta, label=f"Mean Accuracy Delta {mean_delta}", linestyle='--', color='black')
                                classes_tick = [str(i) if i not in open_classes else str(i)+"*" for i in classes]
                                plt.xticks(classes, classes_tick)
                                plt.xlabel('Class label (* if in open class, red bar if in initial labeled set, green if newly discovered)')
                                plt.ylabel('Number of samples for each class')
                                plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
                                # plt.legend()
                                plt.savefig(os.path.join(curr_query_result_path, pool_name+"_result.png"))
                                plt.close('all')
                                print(os.path.join(curr_query_result_path, pool_name+"_result.png"))
                                # Now re-evaluate on test set to see per-sample accuracy
                                # self.eval_closed_set(test_result, trainset_info, dataset_factory)
                                    
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
                                plt.axhline(y=overall_acc, label=f"Overall Test Accuracy {overall_acc}", linestyle='--', color='black')
                                plt.axhline(y=avg_per_class_acc, label=f"Avg Per-Class Accuracy {avg_per_class_acc}", linestyle='-', color='green')
                                classes_tick = [str(i) if i not in open_classes else str(i)+"*" for i in classes]
                                classes_color = ['red' if i in initial_discovered_classes else 'green' for i in classes]
                                plt.bar(classes, classes_acc, align='center', color=classes_color)
                                plt.xticks(classes, classes_tick)
                                plt.xlabel('Class label (* if in open class, red bar if in initial labeled set, green if newly discovered)')
                                plt.ylabel('Per-class Accuracy')
                                plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
                                plt.legend()
                                plt.savefig(os.path.join(curr_query_result_path, "per_class_acc.png"))
                                plt.close('all')

                                with open(os.path.join(curr_query_result_path, "per_class_acc.txt"), "w+") as file:
                                    result_lists = []
                                    for class_i in classes:
                                        if class_i not in open_classes:
                                            if class_i in initial_discovered_classes:
                                                class_str = str(class_i) + " (initial labeled set)"
                                            else:
                                                class_str = str(class_i) + " (newly discovered)"
                                        else:
                                            class_str = str(class_i) + " (open)"
                                        result_lists.append([f"{class_str}", f"{all_test_classes_acc[class_i]}"])
                                    file.write(tabulate(result_lists, headers=['Class Index', 'Per-Class Accuracy/Recall'], tablefmt='orgtbl'))

                                # Calculate the per-class precision (: pred as car & real car/pred as cars)
                                all_test_classes_precision = {i : None for i in set(gt) }
                                for class_i in all_test_classes_precision:
                                    if (pred == class_i).sum() > 0:
                                        class_i_precision = (gt == class_i)[pred == class_i].sum() / (pred == class_i).sum()
                                    else:
                                        class_i_precision = 0
                                    all_test_classes_precision[class_i] = class_i_precision
                                avg_per_class_precision = float(np.array([all_test_classes_precision[i] for i in all_test_classes_precision]).mean())
                                plt.figure(figsize=(20,12))
                                axes = plt.gca()
                                # axes.set_ylim([-1,1])
                                plt.title(f'Per-class precision (i.e. (Pred as car && is car)/(Pred as car)) for budget {b}.')
                                classes = sorted(list(all_test_classes_precision.keys()))
                                classes_precision = [all_test_classes_precision[i] for i in all_test_classes_precision]
                                classes_color = ['red' if i in initial_discovered_classes else 'g' for i in classes]
                                plt.bar(classes, classes_precision, align='center', color=classes_color)
                                plt.axhline(y=avg_per_class_precision, label=f"Avg Per-Class Precision {avg_per_class_precision}", linestyle='-', color='green')
                                classes_tick = [str(i) for i in classes]
                                plt.xticks(classes, classes_tick)
                                plt.xlabel('Class label (* if in open class, red bar if in initial labeled set, green if newly discovered)')
                                plt.ylabel('Per-class Precision')
                                plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
                                plt.legend()
                                plt.savefig(os.path.join(curr_query_result_path, "per_class_precision.png"))
                                plt.close('all')

                                with open(os.path.join(curr_query_result_path, "per_class_precision.txt"), "w+") as file:
                                    result_lists = []
                                    for class_i in classes:
                                        if class_i not in open_classes:
                                            if class_i in initial_discovered_classes:
                                                class_str = str(class_i) + " (initial labeled set)"
                                            else:
                                                class_str = str(class_i) + " (newly discovered)"
                                        else:
                                            class_str = str(class_i) + " (open)"
                                        result_lists.append([f"{class_str}", f"{all_test_classes_precision[class_i]}"])
                                    file.write(tabulate(result_lists, headers=['Class Index', 'Per-Class Precision'], tablefmt='orgtbl'))
                                
    
    def _get_exp_name(self, data_config, training_method, query_method, b, data_rand_seed, silent=False):
        script_prefix = (f"python train.py {self.data} --data_download_path {self.data_download_path} --data_save_path {self.data_save_path} --data_rand_seed {data_rand_seed}"
                        f" --data_config {data_config} --training_method {training_method} --train_mode {self.train_mode} --trainer_save_dir {self.trainer_save_dir}"
                        f" --query_method {query_method} --budget {b}"
                        f" --verbose {str(not silent)}")
        return script_prefix

    def _get_dataset_info(self):
        from utils import get_trainset_info_path
        trainset_info = torch.load(get_trainset_info_path(self.dataset_save_path, self.data))
        total_query_sample_size = len(trainset_info.query_samples)
        
        if self.data in ['CIFAR100', 'CUB200', 'Cars', 'CIFAR10']:
            regular_init_sample_size = DATASET_CONFIG_DICT[self.data]['regular']['num_init_classes'] * DATASET_CONFIG_DICT[self.data]['regular']['sample_per_class']
            fewer_init_sample_size = DATASET_CONFIG_DICT[self.data]['fewer_class']['num_init_classes'] * DATASET_CONFIG_DICT[self.data]['fewer_class']['sample_per_class']
            assert fewer_init_sample_size == DATASET_CONFIG_DICT[self.data]['fewer_sample']['num_init_classes'] * DATASET_CONFIG_DICT[self.data]['fewer_sample']['sample_per_class']
        return total_query_sample_size, regular_init_sample_size, fewer_init_sample_size, list(map(float, self.budget_mode.split("_")))
 

    def _get_budget_candidates(self, analysis_mode=None):
        """Returns:
            budget_list_regular : List of budget for regular setting
            budget_list_fewer : List of budget for fewer class/sample setting
            sample_diff : The difference between the number of starting samples
        """
        assert analysis_mode in ['same_sample', 'same_budget']
        from utils import get_trainset_info_path
        trainset_info = torch.load(get_trainset_info_path(self.dataset_save_path, self.data))
        total_query_sample_size = len(trainset_info.query_samples)
        
        if self.data in ['CIFAR100', 'CUB200', 'Cars', 'CIFAR10']:
            regular_init_sample_size = DATASET_CONFIG_DICT[self.data]['regular']['num_init_classes'] * DATASET_CONFIG_DICT[self.data]['regular']['sample_per_class']
            fewer_init_sample_size = DATASET_CONFIG_DICT[self.data]['fewer_class']['num_init_classes'] * DATASET_CONFIG_DICT[self.data]['fewer_class']['sample_per_class']
            assert fewer_init_sample_size == DATASET_CONFIG_DICT[self.data]['fewer_sample']['num_init_classes'] * DATASET_CONFIG_DICT[self.data]['fewer_sample']['sample_per_class']
            
            regular_unlabeled_pool_size = total_query_sample_size - regular_init_sample_size
            fewer_unlabeled_pool_size = total_query_sample_size - fewer_init_sample_size
            sample_diff = regular_init_sample_size - fewer_init_sample_size
            budget_list = list(map(lambda x : float(x) * 0.01 * regular_unlabeled_pool_size,self.budget_mode.split("_")))
            for b in budget_list:
                if not b.is_integer():
                    print(f"{b} is not an integer.")
            budget_list = list(map(int, budget_list))
        else:
            raise NotImplementedError()
        
        if analysis_mode == 'same_budget':
            return budget_list, budget_list
        elif analysis_mode == 'same_sample':
            return budget_list, list(
                                    map(
                                         lambda x: int(min(fewer_unlabeled_pool_size, x + sample_diff)),
                                         budget_list
                                    )
                                )
            


if __name__ == "__main__":
    from config import get_config
    from global_setting import DATASET_CONFIG_DICT
    from utils import prepare_save_dir
    config = get_config()

    # Below are the settings to want to compare
    # DATA_CONFIGS = ['regular', 'fewer_class', 'fewer_sample']
    # TRAINING_METHODS = ['softmax_network', 'cosine_network']
    DATASET_RAND_SEED_LIST = [None, 1, 10, 100, 1000, 2000, 3000]
    ALL_OPEN_METHODS = ['softmax', 'entropy', 'nn', 'nn_cosine', 'openmax']
    if config.analysis_trainer == 'softmax_network':
        # Softmax network
        TRAINING_METHODS = ['softmax_network']
        QUERY_METHODS = ['random', 'entropy', 'softmax', 'uldr', 'coreset']
    elif config.analysis_trainer == 'cosine_network':
        # Cosine network
        TRAINING_METHODS = ['cosine_network']
        QUERY_METHODS = ['random', 'entropy', 'softmax', 'uldr_norm_cosine', 'coreset_norm_cosine']
    elif config.analysis_trainer == 'deep_metric':
        # Cosine network
        TRAINING_METHODS = ['deep_metric']
        QUERY_METHODS = ['random', 'entropy', 'softmax', 'uldr', 'coreset']
    
    # QUERY_METHODS = ['uldr', 'coreset']
    analysis_machine = AnalysisMachine(config.analysis_save_dir,
                                       config.analysis_trainer,
                                       config.budget_mode,
                                       config.data_download_path,
                                       config.data_save_path,
                                       config.trainer_save_dir,
                                       config.data,
                                       DATASET_RAND_SEED_LIST,
                                       TRAINING_METHODS,
                                       config.train_mode,
                                       QUERY_METHODS)
    
    #### Comment out if not running for retraining mode
    analysis_machine.ALL_TRAIN_MODES = ['retrain']

    ### TODO: Comment out
    # analysis_machine.ALL_TRAIN_METHODS = ['softmax_network']
    # analysis_machine.ALL_QUERY_METHODS = ['random']
    # DATASET_RAND_SEED_LIST = [None]
    
    #### Comment out if not answering basic question
    # analysis_machine.ALL_TRAIN_METHODS = ['softmax_network']
    # analysis_machine.PLOT_MODE = ['compare_setting',]
    # analysis_machine.draw_closed_set(draw_open=False)


    #### Comment out if not answering set 1 of basic question
    # analysis_machine.ALL_TRAIN_METHODS = ['softmax_network']
    # analysis_machine.PLOT_MODE = ['compare_setting']
    # analysis_machine.draw_closed_set(draw_open=True)

    #### Comment out if not answering of update rule
    # analysis_machine.ALL_TRAIN_METHODS = ['softmax_network']
    # analysis_machine.ALL_DATA_CONFIGS = ['regular']
    # analysis_machine.ALL_TRAIN_MODES = ['default', 'default_lr01_200eps', 'fix_feature_extractor']
    # analysis_machine.PLOT_MODE = ['compare_active']
    # analysis_machine.draw_train_mode(draw_open=True)

    # analysis_machine.ALL_TRAIN_METHODS = ['softmax_network']
    # analysis_machine.ALL_TRAIN_MODES = ['default_lr01_200eps']
    # analysis_machine.PLOT_MODE = ['compare_active']
    # analysis_machine.ALL_DATA_CONFIGS = ['regular']
    # analysis_machine.draw_closed_set(draw_open=True)

    #### Comment out if not answering of update rule for deep metric
    # analysis_machine.ALL_TRAIN_METHODS = ['deep_metric']
    # analysis_machine.ALL_DATA_CONFIGS = ['regular']
    # analysis_machine.ALL_TRAIN_MODES = ['default', 'default_lr01_200eps']
    # analysis_machine.PLOT_MODE = ['compare_active']
    # analysis_machine.draw_train_mode(draw_open=True)

    #### 
    # Check all checkpoint files exist
    analysis_machine.check_ckpts_exist()
    results = analysis_machine.draw_closed_set(draw_open=True, draw_perclass=True)
    
    