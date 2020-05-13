# logging_helper.py includes several helper functions for
# (1) Converting experiment details into string
#       get_data_param : Dataset info
#       get_owar_param : OWAR learning setup including dataset info
#       get_method_param : Method overview
#       get_active_param : Active query criterion
#       get_experiment_name : Detailed experiment info


def get_data_param(config):
    """Returns dataset (class split) information as str
    """
    return "_".join([config.data, config.init_mode])

def get_owar_param(config):
    """Returns dataset (class split) information
       and open world active recognition (OWAR) setup (rounds/budget/retraining mode) as str
    """
    # For learning loss active learning logging
    name = [config.data, config.init_mode, "r", str(config.max_rounds), "b", str(config.budget)]
    name += ['retrain', str(config.icalr_mode), str(config.icalr_retrain_threshold), str(config.icalr_retrain_criterion)]
    name += ['exemplar', str(config.icalr_exemplar_size)]

    name += ['icalr', str(config.icalr_strategy)]
    if config.icalr_strategy == 'naive':
        name += ['mode', str(config.icalr_naive_strategy)]
    elif config.icalr_strategy == 'smooth':
        name += ['smooth_eps', str(config.smooth_epochs)]
    elif config.icalr_strategy == 'proto':
        name += ['mode', str(config.icalr_proto_strategy)]
    
    return "_".join(name)

def get_method_param(config):
    """Returns method information as str
    """
    # For first round thresholds values logging
    if config.trainer in ['network','icalr','binary_softmax']:
        setting_str = config.threshold_metric
    elif config.trainer == 'sigmoid':
        setting_str = config.sigmoid_train_mode
    elif config.trainer == "icalr_binary_softmax":
        setting_str = config.icalr_binary_softmax_train_mode
    elif config.trainer == 'c2ae':
        setting_str = config.c2ae_train_mode
    elif config.trainer in ['osdn_modified', 'osdn', 'icalr_osdn_modified', 'icalr_osdn', 'icalr_osdn_modified_neg', 'icalr_osdn_neg']:
        setting_str = config.distance_metric
    elif config.trainer in ['cluster']:
        setting_str = "_".join([config.clustering, "dist", config.distance_metric, "metric", config.threshold_metric])
    elif config.trainer in ['network_learning_loss', 'icalr_learning_loss']:
        setting_str = "_".join([config.threshold_metric, 'mode', config.learning_loss_train_mode, 'lmb', str(config.learning_loss_lambda),
                                'margin', str(config.learning_loss_margin),
                                'start_ep', str(config.learning_loss_start_epoch),
                                'stop_ep', str(config.learning_loss_stop_epoch)])
    else:
        raise NotImplementedError()
    return "_".join([config.trainer, setting_str])

def get_active_param(config):
    """Returns active learning criterion as str.
    """
    # For active learning acc logging    
    name = [config.label_picker]
    if config.label_picker == "uncertainty_measure":
        name += [config.uncertainty_measure, "s", config.active_random_sampling]
    elif config.label_picker == "coreset_measure":
        name += [config.coreset_measure, "s", config.active_random_sampling, config.coreset_feature]
    else:
        raise NotImplementedError()
    name += ['oa', config.open_active_setup]
    return "_".join(name)

def get_experiment_name(config):
    """Returns all detailed information of this experiment as str.
    """
    name_str = ''

    name = []
    name += [config.data]
    name += ['rounds', str(config.max_rounds), 'budget', str(config.budget), 'init', config.init_mode]
    name += ['retrain', str(config.icalr_mode), str(config.icalr_retrain_threshold), str(config.icalr_retrain_criterion)]
    name += ['exemplar', str(config.icalr_exemplar_size)]
    name_str += "_".join(name) + os.sep

    name = []
    name += ['icalr', str(config.icalr_strategy)]
    if config.icalr_strategy == 'naive':
        name += ['mode', str(config.icalr_naive_strategy)]
    elif config.icalr_strategy == 'proto':
        name += ['mode', str(config.icalr_proto_strategy)]
    elif config.icalr_strategy == 'smooth':
        name += ['smooth_eps', str(config.smooth_epochs)]
    name_str += "_".join(name) + os.sep
    
    name = []

    if config.label_picker == "uncertainty_measure":
        name += ["uncertain"]
        name += [config.uncertainty_measure, config.active_random_sampling]
    elif config.label_picker == "coreset_measure":
        name += ["coreset"]
        name += [config.coreset_measure, config.active_random_sampling, config.coreset_feature]
    else:
        raise NotImplementedError()
    name += ['oa', config.open_active_setup]
    name_str += "_".join(name) + os.sep

    name = []
    if config.trainer == 'gan':
        name += ['gan', config.gan_player, 'mode', config.gan_mode, 'setup', config.gan_setup]
        if config.gan_player == 'multiple':
            name += ['multi', config.gan_multi]
    elif config.trainer in ["network", 'icalr']:
        name += ['openset', config.threshold_metric, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer in ["network_learning_loss", 'icalr_learning_loss']:
        name += ['learning_loss', config.threshold_metric, config.network_eval_mode, str(config.network_eval_threshold), 
                 # 'mode', config.learning_loss_train_mode, 
                 'lmb', str(config.learning_loss_lambda),
                 'margin', str(config.learning_loss_margin),
                 'start_ep', str(config.learning_loss_start_epoch),
                 'stop_ep', str(config.learning_loss_stop_epoch)]
    elif config.trainer == "sigmoid":
        name += ["sigmoid", config.sigmoid_train_mode, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer == "binary_softmax":
        name += [config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer == "icalr_binary_softmax":
        name += [config.icalr_binary_softmax_train_mode, config.network_eval_mode, str(config.network_eval_threshold)]
    elif config.trainer == "c2ae":
        name += ["c2ae", config.c2ae_train_mode, "alpha", str(config.c2ae_alpha), ]
    elif config.trainer in ['osdn','osdn_modified']:
        name += ['osdn' if not config.trainer == 'osdn_modified' else 'osdnmod', 
                 "dist", config.distance_metric]
        if 'eu' in config.distance_metric:
            name += ['diveu', str(config.div_eu)]
        if config.pseudo_open_set == None:
            # Using fixed hyper
            name += ["thre", str(config.osdn_eval_threshold),
                     "alpha", config.alpha_rank,
                     "tail", config.weibull_tail_size]
        else:
            # Using cross validation/meta learning to decide hyper
            assert config.openmax_meta_learn != None
            name += ["meta", str(config.openmax_meta_learn)]
        # name += ['mav', config.mav_features_selection]
    elif config.trainer in ['icalr_osdn','icalr_osdn_modified', 'icalr_osdn_neg', 'icalr_osdn_modified_neg']:
        name += ['icalr_openmax' if not config.trainer == 'icalr_osdn_modified' else 'icalr_osdnmod', 
                 "dist", config.distance_metric]
        if 'eu' in config.distance_metric:
            name += ['diveu', str(config.div_eu)]
        if config.pseudo_open_set == None:
            # Using fixed hyper
            name += ["thre", str(config.osdn_eval_threshold),
                     "alpha", config.alpha_rank,
                     "tail", config.weibull_tail_size]
        else:
            # Using cross validation/meta learning to decide hyper
            assert config.openmax_meta_learn != None
            name += ["meta", str(config.openmax_meta_learn)]
        # name += ['mav', config.mav_features_selection]
    elif config.trainer == 'cluster':
        name += ['cluster', config.clustering, 'distance', config.distance_metric]
        if 'eu' in config.distance_metric:
            name += ['div_eu', str(config.div_eu)]
        if config.clustering == 'rbf_train':
            name += ['gamma', str(config.rbf_gamma)]
        if config.pseudo_open_set == None:
            name += ['threshold', str(config.cluster_eval_threshold)]
        else:
            name += ['threshold', 'metalearn']
        name += ['metric', config.threshold_metric]
        name += ['level', config.cluster_level]
    else:
        raise NotImplementedError()

    # if config.trainer in ["network", "osdn", "osdn_modified"]:
    name += ['cw', config.class_weight]
    if config.pseudo_open_set != None:
        name += ['p', str(config.pseudo_open_set), 'r', str(config.pseudo_open_set_rounds), 'm', config.pseudo_open_set_metric]
        if config.pseudo_same_network:
            name += ['same_network']
    name_str += "_".join(name) + os.sep

    name = []
    name += ['baseline', config.trainer]
    name += [config.arch, 'pretrained', str(config.pretrained)]
    name += ["lr", str(config.lr), config.optim, "mt", str(config.momentum), "wd", str(config.wd)]
    name += ["epoch", str(config.epochs), "batch", str(config.batch)]
    if config.lr_decay_step != None:
        name += ['lrdecay', str(config.lr_decay_ratio), 'per', str(config.lr_decay_step)]
    name_str += "_".join(name)

    return name_str
