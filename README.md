# tabpfnvsnaiveautoml

Here we compare the TabPFN with Naive AutoML (https://github.com/fmohr/naiveautoml) in terms of accuracy. Check the config file to see which datasets were used.

## Conditions
### Datasets
Only datasets:
- without categorical attributes
- at most 10 classes
- at most 100 features

Datasets in which we had more than 1000 instances were split so that at most 1000 of the instances were used for training (rest for testing).

### TabPFN
used with ensembling of 4

### Naive AutoML
used with a very short HPO phase of 10 (purely random) hyperparameter configurations; no BO etc. is applied.

## Results
### Accuracy

![image](https://user-images.githubusercontent.com/696908/197792620-5a0e8c89-cd8d-4fef-9dda-57871153875e.png)

From this, TabPFN gives really impressive results (at least to me). However, it seems not generally competitive with AutoML tools yet. Things to keep in mind for the cases where TabPFN beats Naive AutoML:
- Naive AutoML has only a very weak support for neural networks (only fully connected, and hardly fine tuned)
- Naive AutoML does not apply any ensembling (except pre-build ensemblers like RandomForests)
- It is unclear whether some of the datasets used here were part of TabPFN's meta-learning phase, which would be an unfair advantage

### Runtimes
What makes TabPFN really exciting is its runtime behavior.
