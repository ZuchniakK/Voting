# VotingAlgorithms
Implementation of selected voting algorithms with a notebook presenting the application.

## Intro

As part of my PhD research I analyzed the impact of the decision-fusion mechanism inside ensemble models.
Voting algorithms themselves are a very interesting family of algorithms.
With the same set of voters, we can get different election results depending on the voting algorithm we use.

This repository contains the implementation of selected voting algorithms *("plurality", "copeland", "borda", "simpson",
"dowdall", "baldwin", "nanson", "bucklin", "stv",)*,
classes that allow you to analyze how individual algorithms affect the results obtained.
In my research, I checked how well the election algorithm is able to extract the truth from a group with noisy information.


[voting_explained.ipynb](voting_explained.ipynb) notebook shows examples of using, generating single and batch voting,
I show examples that for the same set of voter preferences by modifying the used voting algorithm we will get different results, and some interesting statistical properties of the obtained election results.

[src/voting_methods.py](./src/voting_methods.py) - Classes that implement voting algorithms and enable their statistical analysis.
<br />
[src/voting_complexity.py](./src/voting_complexity.py) - Functions that enable testing the computational complexity of the implementation of individual voting algorithms.
<br />
[src/utils.py](./src/utils.py) - Helpers for calculating intermediate values and generating figures.

# Results on MNIST




| Ensemble size | Plurality | Borda  | STV    | Copeland‘s | Softmax |
|---------------|-----------|--------|--------|------------|---------|
| 5             | 58.6\%    | 61.1\% | 58.3\% | 60.0\%     | 62.7\%  |
| 25            | 66.1\%    | 69.8\% | 67.8\% | 68.1\%     | 69.7\%  |
| 55            | 67.7\%    | 71.5\% | 69.6\% | 69.9\%     | 71.1\%  |

Accuracy of classification on MNIST dataset depending on the number of basic
models in the ensemble system. Ensemble submodels are simple fully-connected deep neural networks containing only two
hidden layers with 50 neurons. Each model was trained on a small fraction of the training dataset.

Various voting schemes produce different
dependencies between classification accuracy and the size of the ensemble. Plurality voting gives
the worst results among all the methods tested. The single transferable
vote give slightly better results. 
Borda method generates the highest quality of classification.
Compared to the summing values from the last layer of the neural network (softmax layer), which has much more complete information on the predictions of the submodels (complete vector containing the probabilities of belonging to each class vs.
preferences list), the performance of the Borda method is comparable to the summation of softmax layers for larger values of *N(>20)*.
However, for smaller values of *N*, summing the probabilities (from the softmax layer) is the most efficient method.
This is mainly due to the fact that the preference list loses some of the information returned by individual submodels.
