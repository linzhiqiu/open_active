

# python plot_active.py

# Assuming all thresholds value results are stored in folder first_round_thresholds/
import json, argparse, os
import numpy as np
from glob import glob
from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import pickle

def break_if_too_long(name, split=80):
    if len(name) > split:
        names = [name[x:x+split] for x in range(0, len(name), split)]
        return "\n".join(names)
    else:
        return name

def first_round_seen_all(dictionary, num_discovered_classes=90):
    for round_idx in dictionary.keys():
        return num_discovered_classes == dictionary[round_idx]['num_discovered_classes']

def paper_version_title(name):
    if "our_auroc" in name:
        return "Area Under GOSCR Curve"
    elif "roc_auroc" in name:
        return "Area Under ROC Curve"
    elif "overall_acc" in name:
        return "Overall Accuracy"
    elif "class" in name:
        return "Number of discovered classes"
    elif "discovered_closed_acc" in name:
        return "Closed-set Accuracy on Discovered Class"
    else:
        return name

def paper_version_ylabel(name):
    if "auroc" in name:
        return "Area Under Curve"
    elif "acc" in name:
        return "Accuracy"
    elif "class" in name:
        return "Number of classes"
    else:
        return name

def paper_version(name):
    if "/random_proto/" in name:
        return "Prototypical"
    elif "/random_naive/" in name:
        return "Naive"
    elif "/active_all_learnloss/" in name:
        return "Learning Loss"
    elif "/active_all_learnlossneg/" in name:
        return "Learning Loss (Negative)"
    elif "/active_all_coreset/" in name:
        return "Core Set (Greedy)"
    elif "/random_all/" in name:
        return "Random Query"
    elif "/random100then_coreset/" in name:
        return "Random Query 100 Rounds => Core-Set (Greedy)"
    elif "/random200then_coreset/" in name:
        return "Random Query 200 Rounds => Core-Set (Greedy)"
    elif "/random300then_coreset/" in name:
        return "Random Query 300 Rounds => Core-Set (Greedy)"
    elif "/random200then_learnloss/" in name:
        return "Random Query 200 Rounds => Learning Loss"
    elif "/random100then_learnloss/" in name:
        return "Random Query 100 Rounds => Learning Loss"
    elif "/random300then_learnloss/" in name:
        return "Random Query 300 Rounds => Learning Loss"
    else:
        return name

def plot_roc(round_results, output_folder=None, round_idx=0, printed=True):
    # Discovered v.s. Hold-out open
    gt = np.array(round_results['thresholds']['ground_truth']) # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
    open_scores = np.array(round_results['thresholds']['open_set_score'])
    gt[gt >= 0] = 0
    gt[gt == OPEN_CLASS_INDEX] = 1
    selected_indices = gt != UNDISCOVERED_CLASS_INDEX
    open_scores = open_scores[selected_indices]
    gt = gt[selected_indices]

    # fpr, tpr, _ = roc_curve(gt, open_scores)
    try:
        if np.any(np.isnan(open_scores)):
            print(f"There is {np.sum(np.isnan(open_scores))} NaN values for {output_folder}. Replace them by the mean of remaining scores.")
            open_scores[np.where(np.isnan(open_scores))] = open_scores.nanmean()
        fpr, tpr, _ = roc_curve(gt, open_scores)
        auc_score = roc_auc_score(gt, open_scores)
    except:
        print(f"Wrong AUC!!! for {output_folder}")
        if printed: import pdb; pdb.set_trace()  # breakpoint 3c78e0d4 //
        else:
            return {'auc_score' : 0.}

    parsed_results = {'fpr' : fpr, 'tpr' : tpr, 'auc_score' : auc_score}

    save_path = os.path.join(output_folder, f"roc_auroc_{auc_score:.4f}_round_{round_idx}.png")
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,1])
    plt.title(f'ROC curve plot at round {round_idx}', y=0.96, fontsize=12)
    plt.xlabel("False Positive Rate (Closed set examples classified as open set)", fontsize=12)
    plt.ylabel("True Positive Rate (Open set example classified as open set)", fontsize=12)

    label_name = f"AUC_"+f"{auc_score:.3f}"
    plt.plot(fpr, tpr, label=label_name, linestyle='-')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=1, mode="expand", borderaxespad=0.)
    plt.legend(loc='upper left',
               borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')
    return parsed_results

def plot_our(round_results, output_folder=None, round_idx=0, printed=True):
    # Discovered v.s. Hold-out open
    gt = np.array(round_results['thresholds']['ground_truth'])
    open_predicted = np.array(round_results['thresholds']['open_predicted'])
    open_scores = np.array(round_results['thresholds']['open_set_score'])
    selected_indices = gt != UNDISCOVERED_CLASS_INDEX
    gt = gt[selected_indices]
    open_predicted = open_predicted[selected_indices]
    open_scores = open_scores[selected_indices]
    # At this point, gt's open example have label OPEN_CLASS_INDEX.
    
    sorted_indices = np.argsort(open_scores) # Sort from smallest to largest
    FP = [0] # Number of wrongly classified open set example
    TC = [0] # Number of correctly classified closed set examples
    N = 0 # A counter of correctly classified closed set examples with open score higher than prior threshold
    N_below_threshold = 0 # A counter of correctly classified closed set examples
    threshold = open_scores.min() # We slide the threshold from low to high
    total_corrects = 0.
    if np.any(np.isnan(open_scores)):
        print(f"There is {np.sum(np.isnan(open_scores))} NaN values for {output_folder}. Replace them by the mean of remaining scores.")
        try:
            open_scores[np.where(np.isnan(open_scores))] = open_scores.nanmean()
        except:
            if printed: import pdb; pdb.set_trace()  # breakpoint 33b10776 //
            else:
                return {'auc_score' : 0.}
    
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

    save_path = os.path.join(output_folder, f"our_auroc_{auc_score:.4f}_maxacc_{max_acc:.4f}_round_{round_idx}.png")
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,1])
    plt.title(f'Open set classification rate plot at round {round_idx}', y=0.96, fontsize=12)
    axes.set_xscale('log')
    axes.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("False Positive Rate (Open set examples classified as closed set)", fontsize=12)
    plt.ylabel("Correct Classification Rate (Closed set examples classified into correct class)", fontsize=12)
    y_axis_key = 'tcr'

    label_name = f"AUC_"+f"{auc_score:.3f}"#+"MAXACC_"+str(max_acc)
    plt.plot(FPR, TCR, label=label_name, linestyle='-')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #            ncol=1, mode="expand", borderaxespad=0.)
    plt.legend(loc='upper left',
               borderaxespad=0., fontsize=10)
        
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.close()
    plt.close('all')
    return parsed_results

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

def read_json(output_folder, interval=1, threshold='default', printed=True, max_round=None):
    parsed_results_json_path = output_folder + ".json"
    parsed_results = None
    try:
        # dictionary = json.load(open(json_file, "r"))
        if not os.path.exists(parsed_results_json_path):
            print(f"{parsed_results_json_path} not exists.")
            return None
        with open(parsed_results_json_path, 'rb') as f:
            parsed_results = pickle.load(f)
        if max_round and len(parsed_results.keys()) < max_round:
            import pdb; pdb.set_trace()  # breakpoint fdbc1b48 //
            print(f"Not enough keys. {len(parsed_results.keys())}")
    except:
        print(f"Wrong reading the file {json_file}")
        if printed:
            import pdb; pdb.set_trace()  # breakpoint b5f4d9b0 //
        else:
            return None

    if max_round:
        rounds_key = sorted(list(parsed_results.keys()), key=lambda x: int(x))[:max_round]
        new_parsed_results = {}
        for round_idx in rounds_key:
            new_parsed_results[round_idx] = parsed_results[round_idx]
        parsed_results = new_parsed_results
    return parsed_results

def plot_accumulated_json_results(parsed_results_dict, output_folder=None, max_round=360):
    json_keys = list(parsed_results_dict.keys())

    round_indices = sorted(list(parsed_results_dict[json_keys[0]].keys()), key=lambda x: int(x))[:max_round]
    first_round_idx = round_indices[0]
    plot_items = list(parsed_results_dict[json_keys[0]][first_round_idx].keys())
    x = np.array(round_indices)
    # json_keys = sorted(json_keys, key=lambda k: parsed_results_dict[k][max_round-1][item], reverse=True)
    for item in plot_items:

        if item in ['class_accuracy', 'our_results', 'roc_results']:
            continue
        save_path = os.path.join(output_folder, f"{item}.png")
        
        plt.figure(figsize=(10,10))
        axes = plt.gca()
        # if "num_seen" in item:
        #     json_keys = sorted(json_keys, key=lambda k: first_round_seen_all(parsed_results_dict[k]), reverse=False)
        # else:   
        # TODO: Set axes right
        y_min = min([parsed_results_dict[k][round_idx][item] for round_idx in range(max_round) for k in json_keys])
        y_max = max([parsed_results_dict[k][round_idx][item] for round_idx in range(max_round) for k in json_keys])

        if 'slope' in item:
            axes.set_ylim([y_min,y_max])
        elif 'acc' == item[-3:] or 'auroc' in item:
            axes.set_ylim([0,1])
            # plt.axhline(y=y_min, label=f"Min = {y_min:.4f}", linestyle='--', color='r')
            # plt.axhline(y=y_max, label=f"Max: {y_max:.4f}", linestyle='--', color='g')
            plt.axhline(y=y_max, linestyle='--', color='g')
        else:
            if y_min != y_max:
                axes.set_ylim([y_min,y_max])
        if min(x) != max(x):
            axes.set_xlim([min(x),max(x)])

        # TODO: Set lenged right
        # paper_version_title(item)
        plt.title(f'{paper_version_title(item)}', fontsize=22)
        plt.xlabel("Round Index", fontsize=16)
        plt.ylabel(f"{paper_version_ylabel(item)}", fontsize=16)

        for i, json_file in enumerate(json_keys):
            parsed_results = parsed_results_dict[json_file]
            y = np.zeros_like(x).astype('float')
            for idx, round_idx in enumerate(x):
                y[idx] = parsed_results[round_idx][item]
            label_name = paper_version(json_file)
            plt.plot(x, y, linestyle='-', label=label_name)
        plt.legend(fontsize=16)            
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"save to {save_path}")
        plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--saved_dir', default='open_active_results_new') # Where the log files will be saved
    # parser.add_argument('--saved_dir', default='temp_open_active_results_360') # Where the log files will be saved
    # parser.add_argument('--saved_dir', default='open_active_results') # Where the log files will be saved
    parser.add_argument('--saved_dir', default='open_active_results_graph_new_360') # Where output of this script will be saved
    parser.add_argument('--format', default='.json') # Output file  will be saved at {name[index]}.{args.output_format}
    parser.add_argument('--interval', default=1, type=int) # Plot every [interval] rounds
    parser.add_argument('--threshold', default='default', choices=['default']) # How to pick the open set threshold.
    parser.add_argument('--multi_worker', default=0, type=int) # The number of workers to use
    parser.add_argument('--max_round', default=360, type=int) # The number of workers to use
    args = parser.parse_args()

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

    experiments = {} # All args
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
                    output_folder = os.path.join(args.saved_dir, json_file[json_file.find(os.sep)+1:json_file.rfind(".")])
                    time_str = json_file.split(os.sep)[-1]
                    parsed_results = read_json(output_folder=output_folder, interval=args.interval, threshold=args.threshold, max_round=args.max_round)
                    method_results[json_file] = parsed_results
                    # json_results[time_str] = parsed_results
            #         json_results[json_file] = parsed_results
            # hyper_results[hyper_folder] = json_results
        # method_results[method_folder] = hyper_results
        plot_accumulated_json_results(method_results, output_folder=dataset_folder, max_round=args.max_round)


