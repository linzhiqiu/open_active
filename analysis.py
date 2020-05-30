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

class AnalysisMachine(object):
    """Store all the configs we want to compare
    """
    def __init__(self, analysis_save_dir, analysis_trainer, budget_mode, data_download_path, dataset_save_path, trainer_save_dir, data, dataset_rand_seed, training_method_list, train_mode, query_method_list, open_set_method_list):
        super().__init__()
        self.silent_mode = False
        self.analysis_save_dir = analysis_save_dir
        self.analysis_trainer = analysis_trainer
        self.budget_mode = budget_mode
        self.train_mode = train_mode
        self.save_dir = self.get_save_dir()
        if not os.path.exists(self.save_dir):
            utils.makedirs(self.save_dir)
        else:
            input(f"Already exists: {self.save_dir} . Overwrite? >>")

        self.script_dir = self.get_script_dir()
        self.plot_dir = self.get_plot_dir()

        self.data = data
        self.dataset_rand_seed = dataset_rand_seed

        self.ALL_TRAIN_METHODS = ['softmax_network', 'cosine_network']
        self.ALL_QUERY_METHODS = ['random', 'entropy', 'uldr', 'uldr_norm_cosine', 'coreset', 'coreset_norm_cosine', 'softmax']
        self.ALL_INIT_MODES = ['regular', 'fewer_class', 'fewer_sample']

        self.PLOT_MODE = ['compare_active', 'compare_train', 'compare_setting']
        self.trainer_save_dir = trainer_save_dir
        self.dataset_save_path = dataset_save_path
        self.data_download_path = data_download_path
        
        self.training_method_list = training_method_list
        self.query_method_list = query_method_list
        # self.open_set_method_list = open_set_method_list
        self.open_set_method = 'softmax'

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
    
    def check_ckpts_exist(self):
        for a_mode in ['same_budget', 'same_sample']:
            budget_list_regular, budget_list_fewer= self._get_budget_candidates(analysis_mode=a_mode)
            
            print("For regular setup, the budgets to query are: " + str(budget_list_regular))
            print("For fewer class/sample setup, the budgets to query are: " + str(budget_list_fewer))
            
            print(f"Saving all unfinished experiments to {self.script_dir}")
            undone_exp = []
            script_file = os.path.join(self.script_dir, f"scripts_{a_mode}.sh")
            script_err = os.path.join(self.script_dir, "scripts.err")
            script_out = os.path.join(self.script_dir, "scripts.out")

            if a_mode == 'same_budget':
                enum_list = [('regular', budget_list_regular),
                             ('fewer_class', budget_list_fewer),
                             ('fewer_sample',budget_list_fewer)]
            elif a_mode == 'same_sample':
                enum_list = [('fewer_class', budget_list_fewer),
                             ('fewer_sample',budget_list_fewer)]
            for init_mode, b_list in enum_list:
                print(f"For {init_mode} setting: The experiments to run are:")
                undone_exp_mode = []
                for b in b_list:
                    undone_exp_b = []
                    b_dir = os.path.join(self.script_dir, init_mode, f"budget_{b}")
                    if not os.path.exists(b_dir): utils.makedirs(b_dir)
                    for training_method in self.training_method_list:
                        for query_method in self.query_method_list:
                            # for open_set_method in self.open_set_method_list:
                            if True:
                                paths_dict = prepare_save_dir(self.dataset_save_path,
                                                            self.data_download_path,
                                                            self.trainer_save_dir,
                                                            self.data,
                                                            init_mode,
                                                            self.dataset_rand_seed,
                                                            training_method,
                                                            self.train_mode,
                                                            query_method,
                                                            b,
                                                            utils.get_open_set_methods(training_method),
                                                            makedir=False)
                                # for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path']:
                                    # for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path', 'open_result_path']:
                                
                                # For all ckpts
                                for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path', 'open_result_paths']:
                                    exp_finished = True
                                    if k == 'open_result_paths':
                                        for o_method in utils.get_open_set_methods(training_method):
                                            if not os.path.exists(paths_dict[k][o_method]):
                                                exp_finished = False
                                    else:
                                        if not os.path.exists(paths_dict[k]):
                                            exp_finished = False
                                    if not exp_finished:
                                        python_script = self._get_exp_name(init_mode,
                                                                            training_method,
                                                                            query_method,
                                                                            b,
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
                    print(f"Mode {init_mode}: {len(undone_exp_mode)} to run.")
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

    def draw_closed_set(self):
        budget_list_regular, budget_list_fewer_same_budget = self._get_budget_candidates(analysis_mode='same_budget')
        _, budget_list_fewer_same_sample = self._get_budget_candidates(analysis_mode='same_sample')
        budget_list_fewer = list(set(budget_list_fewer_same_budget + budget_list_fewer_same_sample))
        budget_list_fewer.sort()
        print("For regular setup, the budgets to query are: " + str(budget_list_regular))
        print("For fewer class/sample setup, the budgets to query are: " + str(budget_list_fewer))
        
        print(f"Saving all unfinished experiments to {self.script_dir}")
        finished_exp = {'regular' : {},
                        'fewer_class' : {},
                        'fewer_sample' : {}}
        finished = 0
        unfinished = 0
        
        for init_mode, b_list in [
                                  ('regular', budget_list_regular),
                                  ('fewer_class', budget_list_fewer),
                                  ('fewer_sample',budget_list_fewer)]:
            print(f"For {init_mode} setting: The experiments completed are:")
            for b in b_list:
                if not b in finished_exp[init_mode]:
                    finished_exp[init_mode][b] = {}
                for training_method in self.ALL_TRAIN_METHODS:
                    if not training_method in finished_exp[init_mode][b]:
                        finished_exp[init_mode][b][training_method] = {}
                    for query_method in self.ALL_QUERY_METHODS:
                        paths_dict = prepare_save_dir(self.dataset_save_path,
                                                        self.data_download_path,
                                                        self.trainer_save_dir,
                                                        self.data,
                                                        init_mode,
                                                        self.dataset_rand_seed,
                                                        training_method,
                                                        self.train_mode,
                                                        query_method,
                                                        b,
                                                        utils.get_open_set_methods(training_method),
                                                        makedir=False)
                        if query_method in finished_exp[init_mode][b][training_method]:
                            import pdb; pdb.set_trace()
                        if os.path.exists(paths_dict['test_result_path']):
                            test_result = torch.load(paths_dict['test_result_path'], map_location=torch.device('cpu'))
                            finished_exp[init_mode][b][training_method][query_method] = test_result
                            finished_exp[init_mode][b][training_method][query_method]['open_results'] = {}
                            for o_method in utils.get_open_set_methods(training_method):
                                if os.path.exists(paths_dict['open_result_paths'][o_method]):
                                        
                                    open_result = torch.load(paths_dict['open_result_paths'][o_method], map_location=torch.device('cpu'))
                                    finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method] = {}
                                    finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method]['auroc'] = open_result['roc']['auc_score']
                                    finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method]['roc'] = open_result['roc']
                                    finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method]['augoscr'] = open_result['goscr']['auc_score']
                                    finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method]['goscr'] = open_result['goscr']
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
            'same_budget' : {'path':os.path.join(self.plot_dir, "same_budget"),
                             'fewer_b_list': budget_list_fewer_same_budget,
                             'regular_b_list': budget_list_regular},
            'combined' : {'path':os.path.join(self.plot_dir, "combined"),
                          'fewer_b_list': budget_list_fewer,
                          'regular_b_list': budget_list_regular},
            
        }
        if not os.path.exists(self.plot_dir): utils.makedirs(self.plot_dir)
        print("All plots are saved at " + self.plot_dir)
        total_pool_size, regular_init_size, fewer_init_size, budget_ratio = self._get_dataset_info()
        
        for k in comparsion_dict.keys():
            for plot_mode in self.PLOT_MODE:
                self._draw_closed_set_plot(plot_mode, finished_exp, k, comparsion_dict, total_pool_size, regular_init_size, fewer_init_size, budget_ratio)
                self._draw_open_set_plot(finished_exp, k, comparsion_dict, total_pool_size, regular_init_size, fewer_init_size, budget_ratio)

    def _draw_closed_set_plot(self, plot_mode, finished_exp, key, comparsion_dict, total_size, regular_size, fewer_size, budget_ratio):
        assert key in comparsion_dict
        path = os.path.join(comparsion_dict[key]['path'], plot_mode)
        if not os.path.exists(path): utils.makedirs(path)
        fewer_b_list = comparsion_dict[key]['fewer_b_list']
        regular_b_list = comparsion_dict[key]['regular_b_list']
        
        for item in ['acc', 'seen']:
            if plot_mode == 'compare_active':
                COMPARARISON = self.ALL_QUERY_METHODS
            elif plot_mode == 'compare_train':
                COMPARARISON = self.ALL_TRAIN_METHODS
            elif plot_mode == 'compare_setting':
                COMPARARISON = self.ALL_INIT_MODES

            for compare_thing in COMPARARISON:
                plt.figure(figsize=(15,12))
                if item == 'acc': plt.title(f'Closed set accuracy'); plt.ylabel(f"Accuracy")
                if item == 'seen': plt.title(f'Class discovered rate'); plt.ylabel(f"Discovered Rate")

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
                axes.set_ylim([0,1])
                axes.set_xlim([0, total_size])
                if key == 'same_sample':
                    plt.xlabel("Number of total labeled samples")
                elif key == 'same_budget':
                    plt.xlabel("Number of budgets after initial round")
                elif key == 'combined':
                    plt.xlabel("Number of total labeled samples")
                
                save_path = os.path.join(path, compare_thing+"_"+item+".png")
                
                ALL_INIT_MODES = self.ALL_INIT_MODES
                ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
                ALL_QUERY_METHODS = self.ALL_QUERY_METHODS
                if plot_mode == 'compare_active':
                    ALL_QUERY_METHODS = [compare_thing]
                elif plot_mode == 'compare_train':
                    ALL_TRAIN_METHODS = [compare_thing]
                else:
                    ALL_INIT_MODES = [compare_thing]

                for init_mode in ALL_INIT_MODES:
                    if init_mode in 'regular':
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
                            y = np.array([None for _ in x]).astype(np.double)
                            for idx, b in enumerate(budget_list):
                                is_ready = False
                                if b in finished_exp[init_mode]:
                                    if training_method in finished_exp[init_mode][b]:
                                        if query_method in finished_exp[init_mode][b][training_method]:
                                            is_ready = True
                                if is_ready:
                                    res = float(finished_exp[init_mode][b][training_method][query_method][item])
                                    y[idx] = res
                            if np.any(np.isfinite(y)):
                                lines += 1
                                label_str = "_".join([init_mode, training_method, query_method])
                                c = color_func(init_mode, training_method, query_method)
                                m = marker_func(init_mode, training_method, query_method)
                                plt.plot(x[np.isfinite(y)],
                                         y[np.isfinite(y)],
                                         label=label_str,
                                         color=c,
                                         marker=m)
                plt.legend()
                
                plt.tight_layout()
                # print(save_path + f"has {lines} lines.")
                plt.savefig(save_path)
                plt.close('all')
        
        ALL_OPEN_METHODS = ['softmax', 'entropy', 'nn', 'nn_cosine', 'openmax']
        for item in ['auroc', 'augoscr']:
            for o_method in ALL_OPEN_METHODS:
                open_path = os.path.join(path, o_method)
                utils.makedirs(open_path)
                if plot_mode == 'compare_active':
                    COMPARARISON = self.ALL_QUERY_METHODS
                elif plot_mode == 'compare_train':
                    COMPARARISON = self.ALL_TRAIN_METHODS
                elif plot_mode == 'compare_setting':
                    COMPARARISON = self.ALL_INIT_MODES

                for compare_thing in COMPARARISON:
                    plt.figure(figsize=(15,12))
                    if item == 'auroc': plt.title(f'Area under ROC'); plt.ylabel(f"Area under curve")
                    if item == 'augoscr': plt.title(f'Area under GOSCR'); plt.ylabel(f"Area under curve")
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
                        plt.xlabel("Number of total labeled samples")
                    elif key == 'same_budget':
                        plt.xlabel("Number of budgets after initial round")
                    elif key == 'combined':
                        plt.xlabel("Number of total labeled samples")
                    
                    save_path = os.path.join(open_path, compare_thing+"_"+item+".png")
                    
                    ALL_INIT_MODES = self.ALL_INIT_MODES
                    ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
                    ALL_QUERY_METHODS = self.ALL_QUERY_METHODS
                    if plot_mode == 'compare_active':
                        ALL_QUERY_METHODS = [compare_thing]
                    elif plot_mode == 'compare_train':
                        ALL_TRAIN_METHODS = [compare_thing]
                    else:
                        ALL_INIT_MODES = [compare_thing]

                    for init_mode in ALL_INIT_MODES:
                        if init_mode in 'regular':
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
                                y = np.array([None for _ in x]).astype(np.double)
                                for idx, b in enumerate(budget_list):
                                    is_ready = False
                                    if b in finished_exp[init_mode]:
                                        if training_method in finished_exp[init_mode][b]:
                                            if query_method in finished_exp[init_mode][b][training_method]:
                                                if o_method in finished_exp[init_mode][b][training_method][query_method]['open_results']:
                                                    is_ready = True
                                    if is_ready:
                                        res = float(finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method][item])
                                        y[idx] = res
                                if np.any(np.isfinite(y)):
                                    lines += 1
                                    label_str = "_".join([init_mode, training_method, "Q:"+query_method, "O:"+o_method])
                                    c = color_func(init_mode, training_method, query_method)
                                    m = marker_func(init_mode, training_method, query_method)
                                    plt.plot(x[np.isfinite(y)],
                                            y[np.isfinite(y)],
                                            label=label_str,
                                            color=c,
                                            linestyle=get_style_func(o_method),
                                            marker=m)
                    plt.legend()
                    
                    plt.tight_layout()
                    # print(save_path + f"has {lines} lines.")
                    plt.savefig(save_path)
                    plt.close('all')
    
    def _draw_open_set_plot(self, finished_exp, key, comparsion_dict, total_size, regular_size, fewer_size, budget_ratio):
        assert key in comparsion_dict
        path = os.path.join(comparsion_dict[key]['path'], "compare_open")
        if not os.path.exists(path): utils.makedirs(path)
        fewer_b_list = comparsion_dict[key]['fewer_b_list']
        regular_b_list = comparsion_dict[key]['regular_b_list']
        
        for item in ['auroc', 'augoscr']:
            ALL_INIT_MODES = self.ALL_INIT_MODES
            ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
            ALL_QUERY_METHODS = self.ALL_QUERY_METHODS

            for init_mode in ALL_INIT_MODES:
                if init_mode in 'regular':
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
                        plt.figure(figsize=(15,12))
                        if item == 'auroc': plt.title(f'Area under ROC'); plt.ylabel(f"Area under curve")
                        if item == 'augoscr': plt.title(f'Area under GOSCR'); plt.ylabel(f"Area under curve")

                        axes = plt.gca()
                        axes.set_ylim([0,1])
                        axes.set_xlim([0, total_size])
                        if key == 'same_sample':
                            plt.xlabel("Number of total labeled samples")
                        elif key == 'same_budget':
                            plt.xlabel("Number of budgets after initial round")
                        elif key == 'combined':
                            plt.xlabel("Number of total labeled samples")
                        for o_method in utils.get_open_set_methods(training_method):
                            y = np.array([None for _ in x]).astype(np.double)
                            for idx, b in enumerate(budget_list):
                                is_ready = False
                                if b in finished_exp[init_mode]:
                                    if training_method in finished_exp[init_mode][b]:
                                        if query_method in finished_exp[init_mode][b][training_method]:
                                            if o_method in finished_exp[init_mode][b][training_method][query_method]['open_results']:
                                                is_ready = True
                                if is_ready:
                                    res = finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method]
                                    y[idx] = float(res[item])
                            if np.any(np.isfinite(y)):
                                label_str = "_".join([init_mode, training_method, "Q:"+query_method, "O:"+o_method])
                                c = get_color_func(o_method)
                                plt.plot(x[np.isfinite(y)],
                                         y[np.isfinite(y)],
                                         label=label_str,
                                         color=c)
                        plt.legend()
                        plt.tight_layout()
                        save_dir = os.path.join(path, init_mode, training_method, query_method)
                        utils.makedirs(save_dir)
                        save_path = os.path.join(save_dir, item+".png")
                        plt.savefig(save_path)
                        plt.close('all')
        
        for item in ['roc', 'goscr']:
            ALL_INIT_MODES = self.ALL_INIT_MODES
            ALL_TRAIN_METHODS = self.ALL_TRAIN_METHODS
            ALL_QUERY_METHODS = self.ALL_QUERY_METHODS

            for init_mode in ALL_INIT_MODES:
                if init_mode in 'regular':
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
                            if item == 'roc': plt.title(f'ROC'); plt.xlabel("False Positive Rate (Closed set examples classified as open set)", fontsize=12); plt.ylabel("True Positive Rate (Open set example classified as open set)", fontsize=12)
                            if item == 'goscr': plt.title(f'GOSCR'); plt.ylabel(f"Area under curve"); plt.xlabel("False Positive Rate (Open set examples classified as closed set)", fontsize=12); plt.ylabel("Correct Classification Rate (Closed set examples classified into correct class)", fontsize=12)

                            axes = plt.gca()
                            axes.set_ylim([0,1])
                            axes.set_xlim([0, 1])

                            for o_method in utils.get_open_set_methods(training_method):
                                is_ready = False
                                if b in finished_exp[init_mode]:
                                    if training_method in finished_exp[init_mode][b]:
                                        if query_method in finished_exp[init_mode][b][training_method]:
                                            if o_method in finished_exp[init_mode][b][training_method][query_method]['open_results']:
                                                is_ready = True
                                if is_ready:
                                    res = finished_exp[init_mode][b][training_method][query_method]['open_results'][o_method][item]
                                    x = res['fpr']
                                    if item == 'roc':
                                        y = res['tpr']
                                    else:
                                        y = res['tcr']
                                        axes.set_xscale('log')
                                        axes.autoscale(enable=True, axis='x', tight=True)
                                        plt.title(f'Open set classification rate plot', y=0.96, fontsize=12)
                                if np.any(np.isfinite(y)):
                                    label_str = "_".join(["O:"+o_method, "AUC:"+str(res['auc_score'])])
                                    c = get_color_func(o_method)
                                    plt.plot(x[np.isfinite(y)],
                                            y[np.isfinite(y)],
                                            label=label_str,
                                            color=c)
                            plt.legend(loc='upper left',
                                    borderaxespad=0., fontsize=10)

                            plt.tight_layout()
                            save_dir = os.path.join(path, init_mode, training_method, query_method, f"budget_{b}")
                            utils.makedirs(save_dir)
                            save_path = os.path.join(save_dir, item+".png")
                            plt.savefig(save_path)
                            plt.close('all')

    def _get_exp_name(self, init_mode, training_method, query_method, b, silent=False):
        script_prefix = (f"python train.py {self.data} --download_path {self.data_download_path} --save_path {self.dataset_save_path} --dataset_rand_seed {self.dataset_rand_seed}"
                        f" --init_mode {init_mode} --training_method {training_method} --train_mode {self.train_mode} --trainer_save_dir {self.trainer_save_dir}"
                        f" --query_method {query_method} --budget {b}"
                        f" --verbose {str(not silent)}")
        return script_prefix

    def _get_dataset_info(self):
        from utils import get_trainset_info_path
        trainset_info = torch.load(get_trainset_info_path(self.dataset_save_path, self.data))
        total_query_sample_size = len(trainset_info.query_samples)
        
        if self.data in ['CIFAR100', 'CUB200', 'Cars']:
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
        
        if self.data in ['CIFAR100', 'CUB200', 'Cars']:
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
    # INIT_MODES = ['regular', 'fewer_class', 'fewer_sample']
    # TRAINING_METHODS = ['softmax_network', 'cosine_network']
    
    if config.analysis_trainer == 'softmax_network':
        # Softmax network
        TRAINING_METHODS = ['softmax_network']
        QUERY_METHODS = ['random', 'entropy', 'softmax', 'uldr', 'coreset']
    elif config.analysis_trainer == 'cosine_network':
        # Cosine network
        TRAINING_METHODS = ['cosine_network']
        QUERY_METHODS = ['random', 'entropy', 'softmax', 'uldr_norm_cosine', 'coreset_norm_cosine']
    
    if config.train_mode == 'no_finetune':
        print("No finetune experiments! Only test random query.")
        import pdb; pdb.set_trace()
        QUERY_METHODS = ['random']

    # QUERY_METHODS = ['uldr', 'coreset']
    OPEN_SET_METHODS = ['softmax'] # TODO: Add candidates
    analysis_machine = AnalysisMachine(config.analysis_save_dir,
                                       config.analysis_trainer,
                                       config.budget_mode,
                                       config.download_path,
                                       config.save_path,
                                       config.trainer_save_dir,
                                       config.data,
                                       config.dataset_rand_seed,
                                       TRAINING_METHODS,
                                       config.train_mode,
                                       QUERY_METHODS,
                                       OPEN_SET_METHODS)
    
    # Check all checkpoint files exist
    analysis_machine.check_ckpts_exist()
    analysis_machine.draw_closed_set()
    exit(0)
    # Output a folder for each json file
    # Require folder structure:
        # DATASET INFO
            # METHOD 1
                # HYPER 1
                    # TIME1.JSON
                    # TIME2.JSON
            # METHOD 2 ..
    # For each method, plot all curves under METHOD folder
    # For all methods comparison, collect best performance under DATASET folder
    DATASET_LISTS = glob(os.path.join(args.saved_dir, "*/"))
    print(f"Parse {len(DATASET_LISTS)} datasets.")

    experiments = [] # All args
    for dataset_folder in DATASET_LISTS:
        METHODS_LIST = glob(os.path.join(dataset_folder, "*/"))
        method_results = {}

        for method_folder in METHODS_LIST:
            HYPERS_LIST = glob(os.path.join(method_folder, "*/"))
            hyper_results = {}
            
            for hyper_folder in HYPERS_LIST:
                JSONS_LIST = glob(os.path.join(hyper_folder, f"*{args.format}"))
                json_results = {}
                for json_file in JSONS_LIST:
                    output_folder = os.path.join(args.out_dir, json_file[json_file.find(os.sep)+1:json_file.rfind(".")])
                    time_str = json_file.split(os.sep)[-1]
                    if args.multi_worker == 0:
                        parsed_results = plot_json(json_file, output_folder=output_folder, interval=args.interval, threshold=args.threshold, max_round=args.max_round)
                    else:
                        experiments.append([json_file, output_folder])
                    # json_results[time_str] = parsed_results


                # if len(list(json_results.keys())) == 0:
                #     # import pdb; pdb.set_trace()  # breakpoint 07109789 //
                #     continue

                # plot_curves(json_results, folder=hyper_folder, sorted_key=args.sorted)
                # save_scores(json_results, folder=hyper_folder, sorted_key=args.sorted)
                # sorted_keys_time = sorted(list(json_results.keys()), key=lambda x: json_results[x][args.sorted])
                # best_json_result = json_results[sorted_keys_time[-1]]

                # hyper_str = hyper_folder.split(os.sep)[-2]
                # hyper_results[hyper_str] = best_json_result

            # if len(list(hyper_results.keys())) == 0:
            #     import pdb; pdb.set_trace()  # breakpoint 185c527b //
            #     continue

            # plot_curves(hyper_results, folder=method_folder, sorted_key=args.sorted)
            # save_scores(hyper_results, folder=method_folder, sorted_key=args.sorted)
            # sorted_keys_hyper = sorted(list(hyper_results.keys()), key=lambda x: hyper_results[x][args.sorted])

            # best_hyper_result = hyper_results[sorted_keys_hyper[-1]]

            # method_str = method_folder.split(os.sep)[-2]
            # method_results[method_str] = best_hyper_result

        # if len(list(method_results.keys())) == 0:
        #     import pdb; pdb.set_trace()  # breakpoint 185c527b //
        #     continue

        # plot_curves(method_results, folder=dataset_folder, sorted_key=args.sorted)
        # save_scores(method_results, folder=dataset_folder, sorted_key=args.sorted)
        # sorted_keys_method = sorted(list(method_results.keys()), key=lambda x: method_results[x][args.sorted])

        # best_method_result = method_results[sorted_keys_method[-1]]

        # data_str = dataset_folder.split(os.sep)[-2]
        # print(f"Best method for dataset {data_str} is {sorted_keys_method[-1]} that achieves {best_method_result[args.sorted]} {args.sorted} score.")
    if args.multi_worker > 0:
        # Run multiproecssing
        from multiprocessing import Pool

        def f(lst):
            json_file, output_folder = lst
            plot_json(json_file, output_folder=output_folder, interval=args.interval, threshold=args.threshold, printed=False, max_round=args.max_round)

        with Pool(args.multi_worker) as p:
            p.map(f, experiments)


