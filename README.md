# TabPFN vs. Naive AutoML

Here we compare the TabPFN with Naive AutoML (https://github.com/fmohr/naiveautoml) in terms of accuracy.
Naive AutoML is a highly competitive AutoML tool, which usually outperforms vanilla auto-sklearn in the sense that good solutions are found (much) faster.

## Conditions
### Datasets
A selection of 81 datasets we used for LCDB (https://github.com/fmohr/lcdb), which comply with the restrictions imposed by TabPFN:
- without categorical attributes
- at most 10 classes
- at most 100 features

Check the config file (or result logs below) to see which datasets were used.

Datasets in which we had more than 1000 instances were split so that at most 1000 of the instances were used for training (rest for testing).

### TabPFN
used with ensembling of 4

### Naive AutoML
used with a very short HPO phase of 10 (purely random) hyperparameter configurations; no BO etc. is applied. No ensembles.

In this setup, it is basically a *greedy algorithm selection* + tiny HPO phase.

## Results
### Test Accuracy

![image](https://user-images.githubusercontent.com/696908/197792620-5a0e8c89-cd8d-4fef-9dda-57871153875e.png)

From this, TabPFN gives really impressive results (at least to me). However, it seems not generally competitive with AutoML tools yet. Things to keep in mind for the cases where TabPFN beats Naive AutoML:
- Naive AutoML has only a very weak support for neural networks (only fully connected, and hardly fine tuned)
- Naive AutoML does not apply any ensembling (except pre-build ensemblers like RandomForests)
- It is unclear whether some of the datasets used here were part of TabPFN's meta-learning phase, which would be an unfair advantage

### Accuracies over Runtime
What makes TabPFN really exciting is its runtime behavior.
However, since most of these datasets are small, runtime is also quite short for Naive AutoML most of the times.
Here, we have the plots only for seed = 0, the other plots are also in the repository.
Also, recall that Naive AutoML is a very simple (even though strong and usually better than auto-sklearn) baseline for AutoML.
So, despite its surprisingly strong performance, TabPFN has still quite some way to go to be able to claim to reach or improve over SOTA.

![image](plots/14-0.png)
![image](plots/16-0.png)
![image](plots/18-0.png)
![image](plots/22-0.png)
![image](plots/28-0.png)
![image](plots/30-0.png)
![image](plots/32-0.png)
![image](plots/44-0.png)
![image](plots/54-0.png)
![image](plots/60-0.png)
![image](plots/181-0.png)
![image](plots/182-0.png)
![image](plots/354-0.png)
![image](plots/679-0.png)
![image](plots/715-0.png)
![image](plots/722-0.png)
![image](plots/727-0.png)
![image](plots/728-0.png)
![image](plots/734-0.png)
![image](plots/735-0.png)
![image](plots/737-0.png)
![image](plots/740-0.png)
![image](plots/751-0.png)
![image](plots/752-0.png)
![image](plots/761-0.png)
![image](plots/772-0.png)
![image](plots/797-0.png)
![image](plots/799-0.png)
![image](plots/803-0.png)
![image](plots/807-0.png)
![image](plots/816-0.png)
![image](plots/821-0.png)
![image](plots/822-0.png)
![image](plots/823-0.png)
![image](plots/833-0.png)
![image](plots/837-0.png)
![image](plots/845-0.png)
![image](plots/846-0.png)
![image](plots/847-0.png)
![image](plots/849-0.png)
![image](plots/866-0.png)
![image](plots/871-0.png)
![image](plots/901-0.png)
![image](plots/903-0.png)
![image](plots/904-0.png)
![image](plots/912-0.png)
![image](plots/914-0.png)
![image](plots/917-0.png)
![image](plots/971-0.png)
![image](plots/976-0.png)
![image](plots/977-0.png)
![image](plots/979-0.png)
![image](plots/980-0.png)
![image](plots/995-0.png)
![image](plots/1019-0.png)
![image](plots/1020-0.png)
![image](plots/1021-0.png)
![image](plots/1049-0.png)
![image](plots/1050-0.png)
![image](plots/1059-0.png)
![image](plots/1067-0.png)
![image](plots/1068-0.png)
![image](plots/1120-0.png)
![image](plots/1464-0.png)
![image](plots/1475-0.png)
![image](plots/1487-0.png)
![image](plots/1489-0.png)
![image](plots/1494-0.png)
![image](plots/23517-0.png)
![image](plots/40497-0.png)
![image](plots/40685-0.png)
![image](plots/40900-0.png)
![image](plots/40982-0.png)
![image](plots/40983-0.png)
![image](plots/40984-0.png)
![image](plots/41027-0.png)
![image](plots/41146-0.png)
![image](plots/41150-0.png)
![image](plots/41156-0.png)
![image](plots/41168-0.png)
![image](plots/41946-0.png)
