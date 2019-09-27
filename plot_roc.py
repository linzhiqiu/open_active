# Assuming all thresholds value results are stored in folder first_round_thresholds/
import json, argparse, os
import numpy as np
from glob import glob
from global_setting import OPEN_CLASS_INDEX, UNSEEN_CLASS_INDEX
from sklearn.metrics import roc_curve, roc_auc_score
# import pandas as pd
# from imutil import show

# Hack to keep matplotlib from crashing when run without X
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Apply sane defaults to matplotlib
# import seaborn as sns
# sns.set_style('darkgrid')


def break_if_too_long(name, split=80):
    if len(name) > split:
        names = [name[x:x+split] for x in range(0, len(name), split)]
        return "\n".join(names)
    else:
        return name

def plot_curves(results, folder=None, mode='roc', open_set='hold_out', sorted_key='auc_score'):
    assert folder != None
    save_path = os.path.join(folder, mode+"_"+open_set+".png")
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,1])
    plt.title(f'{mode} curve plot')
    if mode == 'roc':
        plt.xlabel("False Positive Rate (Closed set examples classified as open set)")
        plt.ylabel("True Positive Rate (Open set example classified as open set)")
        y_axis_key = 'tpr'
    elif mode in ['oscr','zhiqiu']:
        axes.set_xscale('log')
        axes.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel("False Positive Rate (Open set examples classified as closed set)")
        plt.ylabel("Correct Classification Rate (Closed set examples classified into correct class)")
        y_axis_key = 'tcr'

    sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)
    for key in sorted_keys:
        # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
        label_name = key+f"_{sorted_key}_"+str(results[key][sorted_key])
        new_label_name = break_if_too_long(label_name, split=80)
        plt.plot(results[key]['fpr'], results[key][y_axis_key], label=new_label_name, linestyle='-')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0.)
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Fig save to {save_path}")

def save_scores(results, folder=None, mode='roc', open_set='hold_out', sorted_key="auc_score"):
    assert folder != None
    save_path = os.path.join(folder, mode+"_"+open_set+".txt")
    with open(save_path, "w+") as file:
        file.write(f"{sorted_key}|setting_str\n")
        sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)
        for key in sorted_keys:
            file.write(f"{results[key][sorted_key]}|{key}\n")

def calc_auc_score(x, y):
    # x and y should be bounded between [0,1]
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        area = area.dtype.type(area)
    return area


def parse_json(json_file, mode='roc', open_set='hold_out'):
    dictionary = json.load(open(json_file, "r"))
    if mode == 'roc':
        gt = np.array(dictionary['ground_truth'])
        open_scores = np.array(dictionary['open_set_score'])
        gt[gt >= 0] = 0
        if open_set == 'all':
            gt[gt == OPEN_CLASS_INDEX] = 1
            gt[gt == UNSEEN_CLASS_INDEX] = 1
        elif open_set == 'hold_out':
            gt[gt == OPEN_CLASS_INDEX] = 1
            selected_indices = gt != UNSEEN_CLASS_INDEX
            open_scores = open_scores[selected_indices]
            gt = gt[selected_indices]
        else:
            gt[gt == UNSEEN_CLASS_INDEX] = 1
            selected_indices = gt != OPEN_CLASS_INDEX
            open_scores = open_scores[selected_indices]
            gt = gt[selected_indices]

        fpr, tpr, thresholds = roc_curve(gt, open_scores)
        auc_score = roc_auc_score(gt, open_scores)
        parsed_results = {'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'auc_score' : auc_score}
        # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
        # print("Saving ROC scores to file {}".format(options['roc_output']))
        # np.save(options['roc_output'], (fpr, tpr))
    elif mode == 'oscr':
        gt = np.array(dictionary['ground_truth'])
        closed_predicted = np.array(dictionary['closed_predicted'])
        closed_argmax_prob = np.array(dictionary['closed_argmax_prob'])
        if open_set == 'all':
            gt[gt == UNSEEN_CLASS_INDEX] = OPEN_CLASS_INDEX
        elif open_set == 'hold_out':
            selected_indices = gt != UNSEEN_CLASS_INDEX
            gt = gt[selected_indices]
            closed_predicted = closed_predicted[selected_indices]
            closed_argmax_prob = closed_argmax_prob[selected_indices]
        else:
            selected_indices = gt != OPEN_CLASS_INDEX
            gt = gt[selected_indices]
            closed_predicted = closed_predicted[selected_indices]
            closed_argmax_prob = closed_argmax_prob[selected_indices]
            gt[gt == UNSEEN_CLASS_INDEX] = OPEN_CLASS_INDEX
        # At this point, gt's open example have label OPEN_CLASS_INDEX.

        sorted_indices = np.argsort(closed_argmax_prob)[::-1] # Sort from largest to smallest
        FP = [0] # Number of wrongly classified open set example
        TC = [0] # Number of correctly classified closed set examples
        N = 0 # A counter of correctly classified closed set examples with argmax score lower than prior threshold
        N_above_threshold = 0 # A counter of correctly classified closed set examples
        threshold = 1. # We slide the threshold from high to low
        total_corrects = 0.
        for idx in sorted_indices:
            gt_label = gt[idx]
            argmax_prob = closed_argmax_prob[idx]
            argmax_label = closed_predicted[idx]
            if gt_label == OPEN_CLASS_INDEX:
                threshold = argmax_prob
                FP.append(FP[-1]+1) # One more open set example wrongly classified
                TC.append(N) # N more correct classified closed example with argmax prob > prior threshold
            else:
                if gt_label == argmax_label:
                    # Correct prediction
                    total_corrects += 1
                    N_above_threshold += 1
                    if argmax_prob < threshold:
                        N = N_above_threshold
        
        num_closed_set = (gt != OPEN_CLASS_INDEX).sum()
        num_open_set = (gt == OPEN_CLASS_INDEX).sum()
        FPR = np.array(FP[1:]).astype(np.float32) / float(num_open_set)
        TCR = np.array(TC[1:]).astype(np.float32) / float(num_closed_set)
        auc_score = calc_auc_score(FPR, TCR)
        max_acc = total_corrects / num_closed_set
        parsed_results = {'fpr' : FPR, 'tcr' : TCR, 'max_acc' : max_acc, 'auc_score' : auc_score}
    elif mode == 'zhiqiu':
        gt = np.array(dictionary['ground_truth'])
        open_predicted = np.array(dictionary['open_predicted'])
        open_scores = np.array(dictionary['open_set_score'])
        if open_set == 'all':
            gt[gt == UNSEEN_CLASS_INDEX] = OPEN_CLASS_INDEX
        elif open_set == 'hold_out':
            selected_indices = gt != UNSEEN_CLASS_INDEX
            gt = gt[selected_indices]
            open_predicted = open_predicted[selected_indices]
            open_scores = open_scores[selected_indices]
        else:
            selected_indices = gt != OPEN_CLASS_INDEX
            gt = gt[selected_indices]
            open_predicted = open_predicted[selected_indices]
            open_scores = open_scores[selected_indices]
            gt[gt == UNSEEN_CLASS_INDEX] = OPEN_CLASS_INDEX
        # At this point, gt's open example have label OPEN_CLASS_INDEX.

        sorted_indices = np.argsort(open_scores) # Sort from smallest to largest
        FP = [0] # Number of wrongly classified open set example
        TC = [0] # Number of correctly classified closed set examples
        N = 0 # A counter of correctly classified closed set examples with open score higher than prior threshold
        N_below_threshold = 0 # A counter of correctly classified closed set examples
        threshold = open_scores.min() # We slide the threshold from low to high
        total_corrects = 0.
        for idx in sorted_indices:
            gt_label = gt[idx]
            curr_open_score = open_scores[idx]
            openpred_label = open_predicted[idx] # For K+1 method, this could be OPEN_CLASS_INDEX
            if gt_label == OPEN_CLASS_INDEX:
                threshold = curr_open_score
                FP.append(FP[-1]+1) # One more open set example wrongly classified
                TC.append(N) # N more correct classified closed example with open score <= prior threshold
            else:
                if gt_label == openpred_label:
                    # Correct prediction
                    total_corrects += 1
                    N_below_threshold += 1
                    if curr_open_score > threshold:
                    # if curr_open_score >= threshold: # TODO: Figure out which one to use
                        N = N_below_threshold
        
        num_closed_set = (gt != OPEN_CLASS_INDEX).sum()
        num_open_set = (gt == OPEN_CLASS_INDEX).sum()
        FPR = np.array(FP[1:]).astype(np.float32) / float(num_open_set)
        TCR = np.array(TC[1:]).astype(np.float32) / float(num_closed_set)
        auc_score = calc_auc_score(FPR, TCR)
        max_acc = total_corrects / num_closed_set
        parsed_results = {'fpr' : FPR, 'tcr' : TCR, 'max_acc' : max_acc, 'auc_score' : auc_score}
    
    # Plot histogram
    gt = np.array(dictionary['ground_truth'])
    open_scores = np.array(dictionary['open_set_score'])
    gt[gt >= 0] = 0
    if open_set == 'all':
        gt[gt == OPEN_CLASS_INDEX] = 1
        gt[gt == UNSEEN_CLASS_INDEX] = 1
    elif open_set == 'hold_out':
        gt[gt == OPEN_CLASS_INDEX] = 1
        selected_indices = gt != UNSEEN_CLASS_INDEX
        open_scores = open_scores[selected_indices]
        gt = gt[selected_indices]
    else:
        gt[gt == UNSEEN_CLASS_INDEX] = 1
        selected_indices = gt != OPEN_CLASS_INDEX
        open_scores = open_scores[selected_indices]
        gt = gt[selected_indices]
    opens = open_scores[gt == 1]
    closeds = open_scores[gt == 0]
    histo_file = json_file[:json_file.rfind(".")] + "_" + open_set + ".png"
    max_score = max(open_scores)
    min_score = min(open_scores)

    bins = np.linspace(min_score, max_score, 100)
    plt.figure(figsize=(10,10))
    plt.hist(opens, bins, alpha=0.5, label='open set')
    plt.hist(closeds, bins, alpha=0.5, label='closed set')
    plt.legend(loc='upper right')
    # plt.show()
    plt.tight_layout()
    plt.savefig(histo_file)
    print(f"Fig save to {histo_file}")

    return parsed_results


        # def process_files(files_to_process,labels,DIR_filename=None,out_of_plot=False):
        #     p=Pool(len(files_to_process))
        #     to_plot=p.map(process_each_file,files_to_process)
        #     p.close()
        #     p.join()
        #     print "Plotting"
        #     u = []
        #     fig, ax = plt.subplots()
        #     for i,(TP,FP,positives,unknowns) in enumerate(to_plot):
        #         ax.plot(FP/unknowns,TP/positives,label=labels[i])
        #         u.append(unknowns)
        #     ax.set_xscale('log')
        #     ax.autoscale(enable=True, axis='x', tight=True)
        #     ax.set_ylim([0,1])
        #     ax.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
        #     ax.set_xlabel('False Positive Rate : Total Unknowns '+str(list(set(u))[0]), fontsize=18, labelpad=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', default='first_round_thresholds') # Where the log files will be saved
    parser.add_argument('--format', default='.json') # Output file  will be saved at {name[index]}.{args.output_format}
    parser.add_argument('--mode', default='roc', choices=['roc', 'oscr', 'zhiqiu'])
    parser.add_argument('--open_set', default='hold_out', choices=['hold_out', 'unseen', 'all']) # what are considered as open set
    # parser.add_argument('--index', default=0, type=int, help='The index of the accuracy')
    args = parser.parse_args()

    # Require folder structure:
        # DATASET INFO
            # METHOD 1
                # HYPER 1
                    # TIME1.JSON
                    # TIME2.JSON
            # METHOD 2 ..
    # For each method, plot all curves under METHOD folder
    # For all methods comparison, collect best performance under DATASET folder
    if args.mode in ['roc', 'oscr', 'zhiqiu']:
        sorted_key = "auc_score" # The higher the better
    else:
        raise NotImplementedError()

    DATASET_LISTS = glob(os.path.join(args.saved_dir, "*/"))
    print(f"Parse {len(DATASET_LISTS)} datasets.")

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
                    time_str = json_file.split(os.sep)[-1]
                    parsed_results = parse_json(json_file, mode=args.mode, open_set=args.open_set)
                    json_results[time_str] = parsed_results

                if len(list(json_results.keys())) == 0:
                    import pdb; pdb.set_trace()  # breakpoint 07109789 //
                    continue

                plot_curves(json_results, folder=hyper_folder, mode=args.mode, open_set=args.open_set, sorted_key=sorted_key)
                save_scores(json_results, folder=hyper_folder, mode=args.mode, open_set=args.open_set, sorted_key=sorted_key)
                sorted_keys_time = sorted(list(json_results.keys()), key=lambda x: json_results[x][sorted_key])
                best_json_result = json_results[sorted_keys_time[-1]]

                hyper_str = hyper_folder.split(os.sep)[-2]
                hyper_results[hyper_str] = best_json_result

            plot_curves(hyper_results, folder=method_folder, mode=args.mode, open_set=args.open_set, sorted_key=sorted_key)
            save_scores(hyper_results, folder=method_folder, mode=args.mode, open_set=args.open_set, sorted_key=sorted_key)
            sorted_keys_hyper = sorted(list(hyper_results.keys()), key=lambda x: hyper_results[x][sorted_key])

            best_hyper_result = hyper_results[sorted_keys_hyper[-1]]

            method_str = method_folder.split(os.sep)[-2]
            method_results[method_str] = best_hyper_result

        plot_curves(method_results, folder=dataset_folder, mode=args.mode, open_set=args.open_set, sorted_key=sorted_key)
        save_scores(method_results, folder=dataset_folder, mode=args.mode, open_set=args.open_set, sorted_key=sorted_key)
        sorted_keys_method = sorted(list(method_results.keys()), key=lambda x: method_results[x][sorted_key])

        best_method_result = method_results[sorted_keys_method[-1]]

        data_str = dataset_folder.split(os.sep)[-2]
        print(f"Best method for dataset {data_str} is {sorted_keys_method[-1]} that achieves {best_method_result[sorted_key]} {sorted_key} score.")



