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

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def best_fit_slope_and_intercept(xs,ys):
    m = (((xs.mean()*ys.mean()) - (xs*ys).mean()) /
         ((xs.mean()*xs.mean()) - (xs*xs).mean()))
    
    b = ys.mean() - m*xs.mean()
    
    return m, b

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

def plot_histo(round_results, output_folder=None, threshold='default', round_idx=0, printed=True):
    # Plot histogram

    gt = np.array(round_results['thresholds']['ground_truth'])
    open_scores = np.array(round_results['thresholds']['open_set_score'])
    gt[gt >= 0] = 0
    gt[gt == OPEN_CLASS_INDEX] = 1
    selected_indices = gt != UNDISCOVERED_CLASS_INDEX
    open_scores = open_scores[selected_indices]
    if np.any(np.isnan(open_scores)):
        try:
            open_scores[np.where(np.isnan(open_scores))] = open_scores.nanmean()
        except:
            print("Computing mean wrong..")
            if printed: import pdb; pdb.set_trace()  # breakpoint 33b10776 //
            else:
                return 0

    gt = gt[selected_indices]
    opens = open_scores[gt == 1]
    closeds = open_scores[gt == 0]
    
    histo_file = os.path.join(output_folder, f"histogram_{round_idx}.png")

    max_score = max(open_scores)
    min_score = min(open_scores)
    if min_score == max_score:
        max_score += 1e-5

    bins = np.linspace(min_score, max_score, 100)
    plt.figure(figsize=(10,10))
    plt.hist(opens, bins, alpha=0.5, label='open set')
    plt.hist(closeds, bins, alpha=0.5, label='closed set')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(histo_file)
    # print(f"Fig save to {histo_file}")
    # plt.close()
    plt.close('all')
    
    if threshold == 'default':
        # Pick threshold such that open set detection accuracy > 0.5
        return sorted(opens)[int(len(opens)/2)]
    else:
        raise NotImplementedError()
    # return threshold

def parse_round_results(round_results, roc_results=None, our_results=None, picked_threshold=None, output_folder=None, round_idx=0):
    # 4: Use the threshold to compute 
        # a: float - overall accuracy
        # b: scatter - closed set accuracy for each discovered class (w.r.t num of samples) (doesn't really need the threshold)
        # c: scatter - fraction of example as open set for each discovered class (w.r.t num of samples)
        # d: float - open set detection accuracy on hold-out open set
        # e: float - open set detection accuracy on unseen open set
        # f: float - closed set accuracy (considering open set) on discovered classes
        # g: float - closed set accuracy (not considering open set) on discovered classes (doesn't really need the threshold)
    assert picked_threshold != None
    parsed_round_results = {}
    # if 'discovered_classes' in round_results.keys() and 'open_classes' in round_results.keys():
    #     discovered_indices = round_results['discovered_classes']
    #     open_indices = round_results['open_classes']
    #     unseen_indices = round_results['undiscovered_classes']
    # else:
    #     num_total_classes = round_results['num_discovered_classes'] + round_results['num_open_classes'] + round_results['num_undiscovered_classes']
    #     open_indices = range(num_total_classes)[-round_results['num_open_classes']:]
    #     discovered_indices = range(num_total_classes)[:round_results['num_discovered_classes']]
    #     unseen_indices = range(num_total_classes)[round_results['num_discovered_classes']:-round_results['num_open_classes']]
    #     assert len(open_indices) + len(discovered_indices) + len(unseen_indices) == num_total_classes

    ground_truth = np.array(round_results['thresholds']['ground_truth'])
    real_labels = np.array(round_results['thresholds']['real_labels'])
    closed_predicted_real = np.array(round_results['thresholds']['closed_predicted_real'])
    open_set_score = np.array(round_results['thresholds']['open_set_score'])

    if 'learningloss_pred_loss' in round_results['thresholds'].keys():
        learningloss_pred_loss = np.array(round_results['thresholds']['learningloss_pred_loss']).flatten()
        actual_loss = np.array(round_results['thresholds']['actual_loss'])
        if len(learningloss_pred_loss) > 0:
            assert len(learningloss_pred_loss) == len(actual_loss)
            
            if len(learningloss_pred_loss) % 2 != 0:
                learningloss_pred_loss = learningloss_pred_loss[:-1]
                actual_loss = actual_loss[:-1]

            shuffle_in_unison_scary(learningloss_pred_loss, actual_loss)
            
            learningloss_pred_loss_part_1 = learningloss_pred_loss[:int(len(actual_loss)/2)]
            learningloss_pred_loss_part_2 = learningloss_pred_loss[int(len(actual_loss)/2):]
            actual_loss_part_1 = actual_loss[:int(len(actual_loss)/2)]
            actual_loss_part_2 = actual_loss[int(len(actual_loss)/2):]
            actual_pred = actual_loss_part_1 >= actual_loss_part_2
            learningloss_pred = learningloss_pred_loss_part_1 >= learningloss_pred_loss_part_2
            learningloss_acc = learningloss_pred == actual_pred
            parsed_round_results['learnloss_open_acc'] = (learningloss_acc).sum() / float(len(learningloss_acc))

        learningloss_pred_loss = np.array(round_results['thresholds']['learningloss_pred_loss']).flatten()[ground_truth >= 0]
        actual_loss = np.array(round_results['thresholds']['actual_loss'])[ground_truth >= 0]
        if len(learningloss_pred_loss) > 0:
            assert len(learningloss_pred_loss) == len(actual_loss)
            
            if len(learningloss_pred_loss) % 2 != 0:
                learningloss_pred_loss = learningloss_pred_loss[:-1]
                actual_loss = actual_loss[:-1]

            shuffle_in_unison_scary(learningloss_pred_loss, actual_loss)
            
            learningloss_pred_loss_part_1 = learningloss_pred_loss[:int(len(actual_loss)/2)]
            learningloss_pred_loss_part_2 = learningloss_pred_loss[int(len(actual_loss)/2):]
            actual_loss_part_1 = actual_loss[:int(len(actual_loss)/2)]
            actual_loss_part_2 = actual_loss[int(len(actual_loss)/2):]
            actual_pred = actual_loss_part_1 >= actual_loss_part_2
            learningloss_pred = learningloss_pred_loss_part_1 >= learningloss_pred_loss_part_2
            learningloss_acc = learningloss_pred == actual_pred
            parsed_round_results['learnloss_discovered_acc'] = (learningloss_acc).sum() / float(len(learningloss_acc))


    if np.any(np.isnan(open_set_score)):
        print(f"There is {np.sum(np.isnan(open_set_score))} NaN values for {output_folder}. Replace them by the mean of remaining scores.")
        open_set_score[np.where(np.isnan(open_set_score))] = open_set_score[np.where(~np.isnan(open_set_score))].mean()
    open_set_pred = open_set_score > picked_threshold # True then open set. False then closed set.

    overall_corrects = (open_set_pred == (ground_truth < 0)) & (real_labels == closed_predicted_real | (ground_truth < 0)) # 1: Open set correct 2: closed set correct on discovered examples
    
    holdout_class_samples = ground_truth == OPEN_CLASS_INDEX
    unseen_class_samples = ground_truth == UNDISCOVERED_CLASS_INDEX
    discovered_class_samples = ground_truth >= 0

    holdout_corrects = open_set_pred[holdout_class_samples]
    unseen_corrects = open_set_pred[unseen_class_samples]
    discovered_closed_corrects = real_labels[discovered_class_samples] == closed_predicted_real[discovered_class_samples]
    discovered_open_corrects = (real_labels[discovered_class_samples] == closed_predicted_real[discovered_class_samples]) & (~open_set_pred[discovered_class_samples])

    # These will be plotted in the end
    parsed_round_results['overall_acc'] = (overall_corrects).sum() / float(len(overall_corrects))
    parsed_round_results['holdout_open_detect_acc'] = (holdout_corrects).sum() / float(len(holdout_corrects))
    parsed_round_results['unseen_open_detect_acc'] = (unseen_corrects).sum() / float(len(unseen_corrects)) if float(len(unseen_corrects)) > 0 else 0
    parsed_round_results['discovered_closed_acc'] = (discovered_closed_corrects).sum() / float(len(discovered_closed_corrects))
    parsed_round_results['discovered_open_acc'] = (discovered_open_corrects).sum() / float(len(discovered_open_corrects))
    parsed_round_results['roc_auroc'] = roc_results['auc_score']
    parsed_round_results['our_auroc'] = our_results['auc_score']
    parsed_round_results['num_discovered_classes'] = round_results['num_discovered_classes']
    parsed_round_results['picked_threshold'] = picked_threshold


    # Calculate learning loss accuracy over discovered classes.

    # Calculate learning loss accuracy over all classes.

    # Draw scatter plot
    seen_samples = np.array(round_results['seen_samples'])
    train_labels = np.array(round_results['train_labels'])[seen_samples]

    discovered_real_labels = real_labels[discovered_class_samples]
    assert len(set(discovered_real_labels)) == round_results['num_discovered_classes']
    discovered_info_dict = {} # Key is real label, value is {'pred_as_open' : number, 'pred_correct' : number}
    for discovered_label in set(discovered_real_labels):
        discovered_info_dict[discovered_label] = {'pred_open' : 0., 'pred_correct' : 0., 'total' : 0., 'train' : 0.}
        class_mask = real_labels == discovered_label
        train_class_mask = train_labels == discovered_label
        discovered_info_dict[discovered_label]['train'] = train_class_mask.sum()
        discovered_info_dict[discovered_label]['total'] = class_mask.sum()
        discovered_info_dict[discovered_label]['pred_open'] = open_set_pred[class_mask].sum()
        discovered_info_dict[discovered_label]['pred_correct'] = (closed_predicted_real[class_mask] == discovered_label).sum()

    # For each class log the accuracy.
    x_class = np.zeros((int(round_results['num_discovered_classes']) + int(round_results['num_undiscovered_classes']))) - 1 # -1 is nondiscovered class
    y_class = np.zeros_like(x_class).astype('float')
    for discovered_label in discovered_info_dict.keys():
        acc = discovered_info_dict[discovered_label]['pred_correct'] / float(discovered_info_dict[discovered_label]['total'] )
        x_class[discovered_label] = discovered_label
        y_class[discovered_label] = acc
    parsed_round_results['class_accuracy'] = (x_class, y_class)

    # Scatterplot
    x = np.array(list(discovered_info_dict.keys()))
    scatter_x_total = np.zeros_like(x).astype('float')
    scatter_y_open = np.zeros_like(x).astype('float')
    scatter_y_correct = np.zeros_like(x).astype('float')
    for i, x_idx in enumerate(x):
        scatter_x_total[i] = discovered_info_dict[x_idx]['train']
        scatter_y_open[i] = discovered_info_dict[x_idx]['pred_open'] / float(discovered_info_dict[x_idx]['total'])
        scatter_y_correct[i] = discovered_info_dict[x_idx]['pred_correct'] / float(discovered_info_dict[x_idx]['total'] )

    save_path_open = os.path.join(output_folder, f"scatterplot_discovered_set_open_fraction_{round_idx}.png")
    save_path_correct = os.path.join(output_folder, f"scatterplot_discovered_set_correct_fraction_{round_idx}.png")
    
    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,max(scatter_x_total)+5])
    plt.title(f'Scatter Plot of Open Set Fraction for discovered classes')
    # axes.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Number of samples in this discovered class")
    plt.ylabel("Fraction wrongly predicted as open set")

    plt.scatter(scatter_x_total, scatter_y_open)
    m_open, b_open = np.polyfit(scatter_x_total, scatter_y_open, 1)
    plt.plot(np.unique(scatter_x_total), np.poly1d((m_open, b_open))(np.unique(scatter_x_total)), label=f"Best Fit Line: y = {m_open:.4} x + {b_open:.4f}", linestyle='-')
    # plt.legend(bbox_to_anchor=(0., 0.97, 1., .102), loc='lower left',borderaxespad=0.)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path_open)
    plt.close('all')

    plt.figure(figsize=(10,10))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,max(scatter_x_total)+5])
    plt.title(f'Scatter Plot of Accuracy for discovered classes (ignore open set detection)')
    # axes.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("Number of samples in this discovered class")
    plt.ylabel("Test Accuracy")

    plt.scatter(scatter_x_total, scatter_y_correct)
    m_correct, b_correct = np.polyfit(scatter_x_total, scatter_y_correct, 1)
    plt.plot(np.unique(scatter_x_total), np.poly1d((m_correct, b_correct))(np.unique(scatter_x_total)), label=f"Best Fit Line: y = {m_correct} x + {b_correct}", linestyle='-')
    plt.axhline(y=parsed_round_results['discovered_closed_acc'], label=f"Mean Accuracy {parsed_round_results['discovered_closed_acc']:.4f}", linestyle='--')
    # plt.legend(bbox_to_anchor=(0., 0.94, 1., .102), loc='lower left', borderaxespad=0.)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path_correct)
    
    # plt.close()
    plt.close('all')

    parsed_round_results['slope_open_set_detection_rate_vs_num_example'] = m_open
    parsed_round_results['slope_closed_set_accuracy_vs_num_example'] = m_correct

    return parsed_round_results
    # self.thresholds_checkpoints[self.round] = {'ground_truth' : [], # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
    #                                            'real_labels' : [], # The real labels for CIFAR100 or other datasets.
    #                                            'open_set_score' : [], # Higher the score, more likely to be open set
    #                                            'closed_predicted' : [], # If fail the open set detection, then what's the predicted closed set label (network output)?
    #                                            'closed_predicted_real' : [], # If fail the open set detection, then what's the predicted closed set label (real labels)?
    #                                            'closed_argmax_prob' : [], # If fail the open set detection, then what's the probability for predicted closed set class (real labels)?
    #                                            'open_predicted' : [], # What is the true predicted label including open set/ for k class method, this is same as above (network predicted output)
    #                                            'open_predicted_real' : [], # What is the true predicted label including open set/ for k class method, this is same as above (real labels)
    #                                            'open_argmax_prob' : [], # What is the probability of the true predicted label including open set/ for k class method, this is same as above
    #                                           } # A list of dictionary

def plot_round(round_results, output_folder, threshold='default', prev_dict=None, prev_round=None, round_idx=0, printed=True):    
    # 1: Plot ROC curve (discovered v.s. hold-out), save fig, get
        # a: float - AUROC
        # b: curve - ROC
    # 2: Plot Our curve (discovered v.s. hold-out), save fig, get
        # a: float - AUROC
        # b: curve - Our
    # 3: Decide the optimal threshold based on threshold argument
        # a: threshold
        # b: Box plot of open scores (discovered v.s. hold-out open)
    # 4: Use the threshold to compute 
        # a: float - overall accuracy
        # b: scatter - closed set accuracy for each discovered class (w.r.t num of samples) (doesn't really need the threshold)
        # c: scatter - fraction of example as open set for each discovered class (w.r.t num of samples)
        # d: float - open set detection accuracy on hold-out open set
        # e: float - open set detection accuracy on unseen open set
        # f: float - closed set accuracy (considering open set) on discovered classes
        # g: float - closed set accuracy (not considering open set) on discovered classes (doesn't really need the threshold)
    # parsed_open_scores = parse_open_scores(round_results) # Use round_results['thresholds']
    # 5: Plot delta accuracy
    # 6: Plot query class accuracy
    roc_results = plot_roc(round_results, output_folder=output_folder, round_idx=round_idx, printed=printed) # {'fpr' : fpr, 'tpr' : tpr, 'auc_score' : auc_score}
    our_results = plot_our(round_results, output_folder=output_folder, round_idx=round_idx, printed=printed) # {'fpr' : FPR, 'tcr' : TCR, 'max_acc' : max_acc, 'auc_score' : auc_score}
    picked_threshold = plot_histo(round_results, output_folder=output_folder, round_idx=round_idx, threshold=threshold, printed=printed) # return picked threshold
    # picked_threshold = pick_threshold(roc_results, our_results, threshold=threshold)
    results = parse_round_results(round_results,
                                  roc_results=roc_results,
                                  our_results=our_results,
                                  picked_threshold=picked_threshold,
                                  output_folder=output_folder,
                                  round_idx=round_idx)
    
    results['roc_results'] = roc_results
    results['our_results'] = our_results
    if type(prev_dict) == type(None) or type(prev_round) == type(None):
        pass
    else:
        # 6: Plot query class accuracy
        query_samples = np.array(list(set(round_results['seen_samples']).difference(prev_dict['seen_samples'])))
        query_classes = set(list(np.array(round_results['train_labels'])[query_samples]))
        y_query = np.zeros_like(results['class_accuracy'][0]).astype('float')
        x_query_ticks = ["" if not i in query_classes else str(i) for i in range(len(y_query))]
        x_query = np.arange(len(y_query))
        for query_class in query_classes:
            y_query[query_class] = results['class_accuracy'][1][query_class]
        
        plt.figure(figsize=(20,12))
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.title(f'Test accuracy for classes being queried in round {round_idx}.')
        plt.bar(x_query, y_query, align='center')
        plt.xticks(x_query, x_query_ticks)
        plt.xlabel('Class label (Those with * are new class just discovered).')
        plt.ylabel('Test accuracy for each class')
        plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
        plt.axhline(y=results['discovered_closed_acc'], label=f"Mean Accuracy {results['discovered_closed_acc']}", linestyle='--', color='black')
        plt.legend()
        save_path_query = os.path.join(output_folder, f"query_class_correct_fraction_{round_idx}.png")
        plt.savefig(save_path_query)
        plt.close('all')

        # 5: Plot delta accuracy
        mean_delta = results['discovered_closed_acc'] - prev_round['discovered_closed_acc']

        x_class, y_class = results['class_accuracy']
        x_class_prev, y_class_prev = prev_round['class_accuracy']
        valid_class = (x_class >= 0) & (x_class_prev >= 0)
        new_class = (x_class >= 0) & (x_class_prev < 0)
        x_delta = np.arange(len(x_class))
        y_delta = np.zeros_like(x_delta).astype('float')
        x_delta_ticks = [str(i)+"*" if new_class[i] else "" if not valid_class[i] else str(i) for i in range(len(x_delta))]
        valid_class_indices = np.where(valid_class)[0]
        y_delta[valid_class_indices] = y_class[valid_class_indices] - y_class_prev[valid_class_indices]
        y_delta_pos = np.zeros_like(y_delta).astype('float')
        y_delta_neg = np.zeros_like(y_delta).astype('float')

        y_delta_pos[y_delta > 0] =  y_delta[y_delta>0]
        y_delta_neg[y_delta < 0] =  y_delta[y_delta<0]

        plt.figure(figsize=(20,12))
        axes = plt.gca()
        axes.set_ylim([-1,1])
        plt.title(f'Delta test accuracy for discovered classes in round {round_idx}.')
        plt.bar(x_delta, y_delta_pos, align='center', color='g')
        plt.bar(x_delta, y_delta_neg, align='center', color='r')
        plt.axhline(y=mean_delta, label=f"Mean Accuracy Delta {mean_delta}", linestyle='--', color='black')
        plt.xticks(x_delta, x_delta_ticks)
        plt.xlabel('Class label')
        plt.ylabel('Delta accuracy for each class: (Current round accuracy - Previous round accuracy)')
        plt.setp(axes.get_xticklabels(), rotation=90, horizontalalignment='right', fontsize='xx-small')
        plt.legend()
        save_path_delta = os.path.join(output_folder, f"delta_class_accuracy_{round_idx}.png")
        plt.savefig(save_path_delta)
        plt.close('all')
    return results

def plot_json(json_file, output_folder, interval=1, threshold='default', printed=True, max_round=None):
    parsed_results_json_path = os.path.join(output_folder, "parsed.json")
    parsed_results = None
    try:
        dictionary = json.load(open(json_file, "r"))
        if os.path.exists(parsed_results_json_path):
            print(f"{parsed_results_json_path} already exists.")
            with open(parsed_results_json_path, 'rb') as f:
                parsed_results = pickle.load(f)
            if len(dictionary.keys()) == len(parsed_results.keys()):
                return parsed_results
            else:
                print(f"But {len(parsed_results.keys())} entries smaller than {len(dictionary.keys())}")
    except:
        print(f"Wrong reading the file {json_file}")
        if printed:
            import pdb; pdb.set_trace()  # breakpoint b5f4d9b0 //
        else:
            return None
    # 1: plot the number of seen classes
    # 2: plot AUROC (ours+ROC) over round
    # 3: plot closed set accuracy over round (discovered+seen) based on the threshold
    # 4: plot closed set accuracy over round (discovered) based on the threshold
    # 5: Plot overall accuracy based on the threshold
    # 6: Plot ROC curve over rounds on a single plot
    sorted_keys = sorted(list(dictionary.keys()), key=lambda x: int(x)) # Sort by round
    if type(parsed_results) == type(None):
        parsed_results = {}
        print(f"Working on {output_folder}")
        start_round = 0
    else:
        print(f"Continue working on {output_folder} from round {len(parsed_results.keys())-1}")
        start_round = len(parsed_results.keys())-1
    output_folder_interval = os.path.join(output_folder, f'plot_every_{args.interval}_round_{threshold}_threshold')
    sorted_keys_tqdm = tqdm.tqdm(sorted_keys) if printed else sorted_keys
    for round_idx in sorted_keys_tqdm:
        if int(round_idx) < start_round:
            continue
        if max_round != None and int(round_idx) > max_round:
            break
        output_folder_round = os.path.join(output_folder_interval, round_idx)
        if not os.path.exists(output_folder_round): os.makedirs(output_folder_round)
        if int(round_idx) == 0:
            round_results = plot_round(dictionary[round_idx], output_folder=output_folder_round, threshold=threshold, prev_dict=None, prev_round=None, round_idx=int(round_idx), printed=printed)
        else:
            round_results = plot_round(dictionary[round_idx], output_folder=output_folder_round, threshold=threshold, prev_dict=dictionary[str(int(round_idx)-1)], prev_round=parsed_results[int(round_idx)-1], round_idx=int(round_idx), printed=printed)
        parsed_results[int(round_idx)] = round_results

    # These are the available gadgets
    # parsed_round_results['overall_acc'] = (overall_corrects).sum() / len(overall_corrects)
    # parsed_round_results['holdout_open_detect_acc'] = (holdout_corrects).sum() / len(holdout_corrects)
    # parsed_round_results['unseen_open_detect_acc'] = (unseen_corrects).sum() / len(unseen_corrects)
    # parsed_round_results['discovered_closed_acc'] = (discovered_closed_corrects).sum() / len(discovered_closed_corrects)
    # parsed_round_results['discovered_open_acc'] = (discovered_open_corrects).sum() / len(discovered_open_corrects)
    # parsed_round_results['roc_auroc'] = roc_results['auc_score']
    # parsed_round_results['our_auroc'] = our_results['auc_score']
    # parsed_round_results['num_discovered_classes'] = our_results['num_discovered_classes']
    # parsed_round_results['threshold'] = picked_threshold
    # Plus the class accuracy tuple (x_class, y_class) each is of size number_total_classes_to_discover
    if max_round:
        rounds_key = sorted_keys[:max_round]
        new_parsed_results = {}
        for round_idx in rounds_key:
            new_parsed_results[int(round_idx)] = parsed_results[int(round_idx)]
        parsed_results = new_parsed_results
    plot_accumulated_rounds(parsed_results, output_folder=output_folder_interval)
    plot_closed_set_accuracy(parsed_results, classes=[0,10], output_folder=output_folder_interval)
    with open(parsed_results_json_path, "wb") as f_out:
        pickle.dump(parsed_results, f_out, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Writed to {parsed_results_json_path}")
    return parsed_results

def plot_accumulated_rounds(parsed_results, output_folder=None):
    round_indices = sorted(list(parsed_results.keys()), key=lambda x: int(x))
    first_round_idx = round_indices[0]
    plot_items = list(parsed_results[first_round_idx].keys())
    x = np.array(round_indices)
    for item in plot_items:
        if item in ['class_accuracy', 'our_results', 'roc_results']:
            continue
        y = np.zeros_like(x).astype('float')
        for idx, round_idx in enumerate(x):
            y[idx] = parsed_results[round_idx][item]
        save_path = os.path.join(output_folder, f"{item}.png")
        plt.figure(figsize=(10,10))
        axes = plt.gca()
        if 'acc' == item[-3:] or 'auroc' in item:
            axes.set_ylim([0,1])
            plt.axhline(y=min(y), label=f"Min = {min(y):.4f}", linestyle='--', color='r')
            plt.axhline(y=max(y), label=f"Max = {max(y):.4f}", linestyle='--', color='g')
        else:
            if min(y) != max(y):
                axes.set_ylim([min(y),max(y)])
        if min(x) != max(x):
            axes.set_xlim([min(x),max(x)])
        plt.title(f'{item}')
        plt.xlabel("Round Index")
        plt.ylabel(f"{item}")
        plt.legend()
        plt.plot(x, y, linestyle='-')            
        plt.tight_layout()
        plt.savefig(save_path)
        # print(f"Fig save to {save_path}")
        # plt.close()
        plt.close('all')

def plot_closed_set_accuracy(parsed_results, classes=[], output_folder=None):
    # if not "class_accuracy" in parsed_results[0]:
    #     print(f"No closed set accuracy plotted at {output_folder}")
    #     return
    round_indices = sorted(list(parsed_results.keys()), key=lambda x: int(x))
    x = np.array(round_indices)
    for class_idx in classes:
        y = np.zeros_like(x).astype('float')
        for idx, round_idx in enumerate(x):
            y[idx] = parsed_results[round_idx]['class_accuracy'][1][class_idx]
        save_path = os.path.join(output_folder, f"class_{class_idx}_closed_set_acc.png")
        plt.figure(figsize=(10,10))
        axes = plt.gca()
        if min(y) != max(y):
            axes.set_ylim([0,1])
        if min(x) != max(x):
            axes.set_xlim([min(x),max(x)])
        plt.axhline(y=min(y), label=f"Min = {min(y):.4f}", linestyle='--', color='r')
        plt.axhline(y=max(y), label=f"Max = {max(y):.4f}", linestyle='--', color='g')
        plt.title(f'Class {class_idx} closed set accuracy')
        plt.xlabel("Round Index")
        plt.ylabel(f"Test Accuracy")
        plt.legend()
        plt.plot(x, y, linestyle='-')            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close('all')

class AnalysisMachine(object):
    """Store all the configs we want to compare
    """
    def __init__(self, analysis_save_dir, analysis_mode, budget_mode, data_download_path, dataset_save_path, trainer_save_dir, data, dataset_rand_seed, training_method_list, train_mode, query_method_list, open_set_method_list):
        super().__init__()
        self.analysis_save_dir = analysis_save_dir
        self.analysis_mode = analysis_mode
        self.budget_mode = budget_mode

        self.save_dir = self.get_save_dir()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            input(f"Already exists: {self.save_dir} . Overwrite? >>")

        self.script_dir = self.get_script_dir()

        self.data = data
        self.dataset_rand_seed = dataset_rand_seed

        self.trainer_save_dir = trainer_save_dir
        self.dataset_save_path = dataset_save_path
        self.data_download_path = data_download_path
        
        self.training_method_list = training_method_list
        self.train_mode = train_mode
        self.query_method_list = query_method_list
        self.open_set_method_list = open_set_method_list

    def get_save_dir(self):
        return os.path.join(self.analysis_save_dir,
                            self.analysis_mode,
                            self.budget_mode)
    
    def get_script_dir(self):
        return os.path.join(self.get_save_dir(),
                            )

    def check_ckpts_exist(self):
        budget_list_regular, budget_list_fewer = self._get_budget_candidates()
        
        print("For regular setup, the budgets to query are: " + str(budget_list_regular))
        print("For fewer class/sample setup, the budgets to query are: " + str(budget_list_fewer))
        
        print(f"Saving all unfinished experiments to {self.script_dir}")
        undone_exp = []
        script_file = os.path.join(self.script_dir, "scripts.sh")
        script_err = os.path.join(self.script_dir, "scripts.err")
        script_out = os.path.join(self.script_dir, "scripts.out")
        for init_mode, b_list in [
                                #   ('regular', budget_list_regular),
                                  ('fewer_class', budget_list_fewer),
                                  ('fewer_sample',budget_list_fewer)]:
            print(f"For {init_mode} setting: The experiments to run are:")
            undone_exp_mode = []
            for b in b_list:
                undone_exp_b = []
                b_dir = os.path.join(self.script_dir, init_mode, f"budget_{b}")
                if not os.path.exists(b_dir): os.makedirs(b_dir)
                for training_method in self.training_method_list:
                    for query_method in self.query_method_list:
                        for open_set_method in self.open_set_method_list:
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
                                                          open_set_method,
                                                          makedir=False)
                            # for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path', 'test_result_path']:
                            for k in ['trained_ckpt_path', 'query_result_path', 'finetuned_ckpt_path']:
                                if not os.path.exists(paths_dict[k]):
                                    python_script = self._get_exp_name(init_mode,
                                                                         training_method,
                                                                         query_method,
                                                                         b,
                                                                         open_set_method)
                                    idx = len(undone_exp_b)
                                    b_err_i = os.path.join(b_dir, f"{idx}.err")
                                    b_out_i = os.path.join(b_dir, f"{idx}.out")
                                    script = python_script + f" >> >(tee -a {b_out_i} >> {script_out}) 2>> >(tee -a {b_err_i} >> {script_err}) \n"
                                    undone_exp_b.append(script)
                                    break
                if undone_exp_b.__len__() > 0:
                    print(f"Budget {b}: {len(undone_exp_b)} experiments to run.")
                    undone_exp_mode = undone_exp_mode + undone_exp_b   
            if undone_exp_mode.__len__() > 0:
                print(f"Mode {init_mode}: {len(undone_exp_mode)} to run.")
                undone_exp = undone_exp + undone_exp_mode
        if undone_exp.__len__() > 0:
            if os.path.exists(script_file):
                input(f"{script_file} already exists. Overwrite >> ")
            if not os.path.exists(b_dir):
                os.makedirs(b_dir)
                print(f"All error will be saved at {script_err}. Details will be saved at {script_dir}")
            with open(script_file, "w+") as file:
                for i, line in enumerate(undone_exp):
                    file.write(line)
        print(f"Budget analysis {self.analysis_mode}: {len(undone_exp)} experiments to run at {script_file}.")

    def _get_exp_name(self, init_mode, training_method, query_method, b, open_set_method):
        script_prefix = (f"python train.py {self.data} --download_path {self.data_download_path} --save_path {self.dataset_save_path} --dataset_rand_seed {self.dataset_rand_seed}"
                        f" --init_mode {init_mode} --training_method {training_method} --train_mode {self.train_mode} --trainer_save_dir {self.trainer_save_dir}"
                        f" --query_method {query_method} --budget {b} --open_set_method {open_set_method}"
                        f" --verbose False")
        return script_prefix

    def _get_budget_candidates(self):
        """Returns:
            budget_list_regular : List of budget for regular setting
            budget_list_fewer : List of budget for fewer class/sample setting
        """
        assert self.analysis_mode in ['same_sample', 'same_budget']
        import torch
        from utils import get_trainset_info_path
        trainset_info = torch.load(get_trainset_info_path(self.dataset_save_path, self.data))
        total_query_sample_size = len(trainset_info.query_samples)
        
        if self.data in ['CIFAR100', 'CUB200']:
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
                    exit(0)
            budget_list = list(map(int, budget_list))
        else:
            raise NotImplementedError()
        
        if self.analysis_mode == 'same_budget':
            return budget_list, budget_list
        elif self.analysis_mode == 'same_sample':
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
    TRAINING_METHODS = ['softmax_network', 'cosine_network']
    # QUERY_METHODS = ['random', 'entropy', 'softmax']
    QUERY_METHODS = ['uldr', 'coreset']
    OPEN_SET_METHODS = ['softmax'] # TODO: Add candidates
    analysis_machine = AnalysisMachine(config.analysis_save_dir,
                                       config.analysis_mode,
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


