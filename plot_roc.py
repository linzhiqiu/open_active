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


def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    df = pd.DataFrame({'x': x, 'y': y})
    plot = df.plot(x='x', y='y')

    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')
    
    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot

def plot_curves(results, folder=None, mode='roc', open_set='hold_out', sorted_key='auc_score'):
    assert folder != None
    save_path = os.path.join(folder, mode+"_"+open_set+".png")
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,1])
    plt.title(f'{mode} curve plot')
    if mode == 'roc':
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

    sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)
    for key in sorted_keys:
        # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
        plt.plot(results[key]['fpr'], results[key]['tpr'], label=key+f"_{sorted_key}_"+str(results[key][sorted_key]), linestyle='-')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=1, mode="expand", borderaxespad=0.)
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Fig save to {save_path}")

def save_scores(results, folder=None, mode='roc', open_set='hold_out', sorted_key="auc_score"):
    assert folder != None
    save_path = os.path.join(folder, mode+"_"+open_set+".txt")
    with open(save_path, "w+") as file:
        file.write(f"{sorted_key}|setting_str")
        sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)
        for key in sorted_keys:
            file.write(f"{results[key][sorted_key]}|{key}\n")


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
            open_scores = np.extract(gt != UNSEEN_CLASS_INDEX, open_scores)
            gt = np.extract(gt != UNSEEN_CLASS_INDEX, gt)
        else:
            gt[gt == UNSEEN_CLASS_INDEX] = 1
            open_scores = np.extract(gt != OPEN_CLASS_INDEX, open_scores)
            gt = np.extract(gt != OPEN_CLASS_INDEX, gt)

        fpr, tpr, thresholds = roc_curve(gt, open_scores)
        auc_score = roc_auc_score(gt, open_scores)
        parsed_results = {'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'auc_score' : auc_score}
        # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
        # print("Saving ROC scores to file {}".format(options['roc_output']))
        # np.save(options['roc_output'], (fpr, tpr))
        return parsed_results
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', default='first_round_thresholds') # Where the log files will be saved
    parser.add_argument('--format', default='.json') # Output file  will be saved at {name[index]}.{args.output_format}
    parser.add_argument('--mode', default='roc', choices=['roc'])
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
    if args.mode == 'roc':
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



