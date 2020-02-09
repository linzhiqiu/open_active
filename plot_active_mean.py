# python plot_active.py

# Assuming all thresholds value results are stored in folder first_round_thresholds/
import json, argparse, os
import numpy as np
from glob import glob
from global_setting import OPEN_CLASS_INDEX, UNSEEN_CLASS_INDEX
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def break_if_too_long(name, split=80):
    if len(name) > split:
        names = [name[x:x+split] for x in range(0, len(name), split)]
        return "\n".join(names)
    else:
        return name

def plot_curves(results, folder=None, sorted_key='final_acc'):
    classes, means, stds = results
    mean_plus_stds = [mean+std for (mean, std) in zip(means, stds)]
    mean_minus_stds = [mean-std for (mean, std) in zip(means, stds)]
    assert folder != None
    save_path = os.path.join(folder, "test_acc_meanstd.png")
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title(f'Test accuracy mean+std curve plot')
    plt.xlabel("Number of samples")
    plt.ylabel("Test accuracy")

    import pdb; pdb.set_trace()  # breakpoint 4de9bc04 //
    sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)

    
    axes.set_xlim([min(mean_minus_stds),max(mean_plus_stds)])
    x = np.array(classes)
    axes.set_xticks(x)

    for key in sorted_keys:
        # plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
        import pdb; pdb.set_trace()  # breakpoint 58b832fe //
        label_name = key+f"_{sorted_key}_"+str(results[key][sorted_key])[:9]
        new_label_name = break_if_too_long(label_name, split=80)

        y = np.array(base_y.copy()).astype(np.double)
        # y = []
        for i, sample_num_key in enumerate(sample_num_entries):
            res = results[key]['acc_dict'][str(sample_num_key)]
            y[i] = res

        plt.plot(x, y, label=new_label_name, linestyle='-', marker='o')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0.)
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Fig save to {save_path}")

def save_scores(results, folder=None, sorted_key="final_acc"):
    import pdb; pdb.set_trace()  # breakpoint 0a9b4d8c //
    assert folder != None
    save_path = os.path.join(folder, f"{sorted_key}_meanstd.txt")
    with open(save_path, "w+") as file:
        file.write(f"{sorted_key}|setting_str\n")
        sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)
        for key in sorted_keys:
            file.write(f"{results[key][sorted_key]}|{key}\n")

def mean_scores(results):
    classes, means, stds = accumulate_acc_dict(results)
    return classes, means, stds
    # sorted_keys = sorted(list(results.keys()), key=lambda x : results[x][sorted_key], reverse=True)
    # for key in sorted_keys:
    #     file.write(f"{results[key][sorted_key]}|{key}\n")

def accumulate_acc_dict(results):
    all_results = np.zeros(
                    (len(results.keys()), len(results[list(results.keys())[0]]['acc_dict'].keys()))
                  )
    for idx, name in enumerate(results.keys()):
        acc_dict_keys = sorted([int(class_num) for class_num in list(results[name]['acc_dict'].keys())])
        if len(acc_dict_keys) != all_results.shape[1]:
            import pdb; pdb.set_trace()  # breakpoint e66b85ac //
        else:
            for jdx, class_num in enumerate(acc_dict_keys):
                all_results[idx, jdx] = results[name]['acc_dict'][str(class_num)]
    means = all_results.mean(0)
    stds = all_results.std(0)
    classes = acc_dict_keys
    return classes, means, stds



def parse_json(json_file, sorted_key='final_acc'):
    try:
        dictionary = json.load(open(json_file, "r"))
    except:
        import pdb; pdb.set_trace()  # breakpoint b5f4d9b0 //

    if sorted_key == 'final_acc':
        lst_of_keys = [int(k) for k in list(dictionary.keys())]
        lst_of_keys.sort()
        parsed_results = {'acc_dict' : dictionary, 'final_acc' : dictionary[str(lst_of_keys[-1])]}
    return parsed_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dir', default='learning_loss') # Where the log files will be saved
    parser.add_argument('--format', default='.json') # Output file  will be saved at {name[index]}.{args.output_format}
    parser.add_argument('--sorted', default='final_acc', choices=['final_acc']) # Sorted by final test accuracy at last round
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
                    parsed_results = parse_json(json_file, sorted_key=args.sorted)
                    json_results[time_str] = parsed_results

                if len(list(json_results.keys())) == 0:
                    # import pdb; pdb.set_trace()  # breakpoint 07109789 //
                    continue

                
                classes, means, stds = mean_scores(json_results)

                hyper_str = hyper_folder.split(os.sep)[-2]
                hyper_results[hyper_str] = [classes, means, stds]
                plot_curves([classes, means, stds], folder=hyper_folder)

            if len(list(hyper_results.keys())) == 0:
                import pdb; pdb.set_trace()  # breakpoint 185c527b //
                continue

            plot_curves(hyper_results, folder=method_folder, sorted_key=args.sorted)
            save_scores(hyper_results, folder=method_folder, sorted_key=args.sorted)
            sorted_keys_hyper = sorted(list(hyper_results.keys()), key=lambda x: hyper_results[x][args.sorted])

            best_hyper_result = hyper_results[sorted_keys_hyper[-1]]

            method_str = method_folder.split(os.sep)[-2]
            method_results[method_str] = best_hyper_result

        if len(list(method_results.keys())) == 0:
            import pdb; pdb.set_trace()  # breakpoint 185c527b //
            continue

        plot_curves(method_results, folder=dataset_folder, sorted_key=args.sorted)
        save_scores(method_results, folder=dataset_folder, sorted_key=args.sorted)
        sorted_keys_method = sorted(list(method_results.keys()), key=lambda x: method_results[x][args.sorted])

        best_method_result = method_results[sorted_keys_method[-1]]

        data_str = dataset_folder.split(os.sep)[-2]
        print(f"Best method for dataset {data_str} is {sorted_keys_method[-1]} that achieves {best_method_result[args.sorted]} {args.sorted} score.")



