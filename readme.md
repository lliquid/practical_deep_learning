## Training - Debugging - Hyperparameter tuning


1. input data normalization can have a huge effect on training speed, usually normalize input variable to have zero mean and standard deviation of one would work well. (https://stats.stackexchange.com/questions/364735/why-does-0-1-scaling-dramatically-increase-training-time-for-feed-forward-an/364776#364776)

2. learning rate is the first variable to tune, to find a good learning rate, observe the shape of the loss curve under different learning rate settings

3. if the loss curve wiggles quite a lot, there are two possible causes: 1) the learning rate is too high 2) the batch size is too small 

4. sanity check on the neural network implementation:
  - overfit a few data samples
  - check if the loss lies in a reasonable range, for example, the cross-entropy loss for cifar10 image classification problem should be -log(1/10)
  - compare analytical gradient to numerical gradient, calculate numerical gradient as f(x+h) - f(x-h) / 2h
  - check if data augmentation is distorting the input too much
  - check if the regularization is too strong and overwhelms the other terms in the loss function
  
5. dead relus can be an issue if the learning rate is too high. 

6. training / validation acc curve helps identify overfitting problems when a model learns the training data too well that it learns details and noises in the training data and it negatively impacts the models generalizability to unseen data.

7. search for good hyperparameters with random search (not grid search), search from coarse to fine ranges, search in log scale e.g. 10^range(-6, 1)

8. form model ensembles for extra performance: model trained with different random initializations, models with different hyperparameters, model at different epoches, etc.

9. SGD + Momentum / Adam 

Ref: https://cs231n.github.io/neural-networks-3/#annealing-the-learning-rate

## Training - Distributed Training - 




