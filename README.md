# open_active
A library for actively performing image recognition experiments in an open world.

1. Active learning for image recognition
2. Open set image recognition
3. Open set active learning for image recognition

# Features

### Implementation of algorithms
This library provides implementation of basic to state-of-the-arts active learning and open set recognition algorithms. They include:

- Open Set Recognition Algorithms
  - **Softmax Threshold**
  - **Entropy Threshold**
  - **Nearest Neighbor Threshold** (both euclidean and cosine distance)
  - **Openmax Threshold**

- Active Learning Algorithms
  - **Random query**
  - **Uncertainty-based methods** including [softmax, entropy]
  - **Coreset Approach** to active learning (greedy version)
  - **ULDR** (unlabeled to labeled density ratio), proposed in "*The Importance of Metric Learning for Robotic Vision: Open Set Recognition and Active Learning*" (ICRA 2019)

We also provide two basic means of training networks:
- **Softmax-based network**
  - A convolutional neural network with a final softmax layer that produces well-defined probability
- **Cosine-based network**
  - Replace the final softmax layer of the CNN by a cosine similarity distance layer. This is more effective with few-shot settings.
  
You may develop your own active learning/open set recognition algorithms by extending the library.

### Settings
We provide the framework to run experiments in three settings:

1. Open set active learning: 
   - This setting is proposed in "*The Importance of Metric Learning for Robotic Vision: Open Set Recognition and Active Learning*".
   - Procedure:
     - The network was first trained on an initial labeled set.
     - Then an active learning algorithm is used together with the trained CNN to query a subset of samples from the unlabeled pool to label.
     - The new labeled samples are added to the initial labeled set.
     - A new network is trained on the new labeled set.
     - The network will be evaluated in an open set test environment, which may contain classes not seen in labeled set. Therefore, a open set method (such as Openmax) will be adopted for rejecting test samples likely in open classes.
   - Please refer to
            ```
                start_open_active_learning.sh
            ```
2. Active learning: 
      - Procedure:
        - The network was first trained on an initial labeled set. This labeled set should contain samples from all existing classes.
        - Use an active learning algorithm to query a subset of samples from the unlabeled pool to label.
        - The new labeled samples are added to the initial labeled set.
        - A new network is trained on the new labeled set.
        - The network will be evaluated on a test set.
      - Please refer to
            ```
                start_active_learning.sh
            ```
3. Open set recognition:
   - Procedure:
        - The network was first trained on an initial labeled set. 
        - The network will be evaluated on a test set, which contains samples from open classes (not seen by the network).
    - Please refer to
            ```
                start_open_learning.sh
            ```

### Metrics
For open set evaluation, we provides two metrics:
1. **ROC** (Receiver Operating Characteristics)
2. **OSCR** (Open Set Classification Rate, proposed in "Reducing network agnostophobia" (NeuralIPS 2018))

We may use area under ROC or OSCR to evaluate the performance of an algorithm in open set scenario.

# Walkthrough
- ```config.py```
  - An argparser for all training/testing configuration shared by all python scripts in this library.
  - You should modify the default arguments if you do not wish to save the outputs to current folder.
- How to start an experiment:
  - Run below three scripts to start an experiment:
    - ```start_open_learning.py```
    - ```start_active_learning.py```
    - ```start_open_active_learning.py```
- How to start a bunch of open set active experiments, e.g. comparing a set of algorithms, and plot the results:
  - Run the below script to obtain a ```scripts.sh``` file for all unfinished experiments:
    - ```open_set_active_analysis.py```
  - You may check ```open_set_active_analysis.sh``` for examples on how to use this script.
- How to plot the results of a bunch of closed set active experiments, e.g. for different active query algorithms.
  - Run the below script
    - ```closed_set_active_analysis.py```
- ```trainer.py```
  - Contains a Trainer class for performing training/querying/testing.
- ```trainer_machine.py```
  - Contains various training methods such as softmax-based and cosine-based networks.
- ```query_machine.py```
  - Contains various active query methods.
- ```eval_machine.py```
  - Contains various open set learning methods.
- ```dataset_factory.py```
  - Prepare datasets for open set/active learning experiments.
- ```trainer_config.py```
  - Configurations for optimizing the network (e.g., hyperparameters).
- ```utils.py```
  - Contains helper methods for all other modules
- ```transform.py```
  - Implements image preprocessing functions.
- ```global_setting.py```
  - Contains some global configuration


Most of the files are well-documented, so I would suggest you to go to each of the files and read the class definition/APIs for a better understanding of this library.