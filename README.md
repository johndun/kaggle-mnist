# Kaggle-MNIST

### mlp1.lua

* Single hidden layer: 576 -> 2048 -> softmax
* No data augmentation
* Single learning rate: 1.0
* Learning rate decay: rate = rate / (1 + n_batches * 1e-6)
* Momentum: 0.9
* Validation set accuracy: 0.97822
* Kaggle leaderboard on 2015-01-11 (accuracy, rank): 0.97957,  127

### mlp2.lua

* Same model as mlp1.lua
* No data augmentation
* Model was trained at 2 learning rates: 1.0 and 0.1
* For the submission model, the number of batches at each learning rate was 
  selected using the validation set results 
* Validation set accuracy: 0.97872
* Kaggle leaderboard on 2015-01-12: 0.97929, 127


### convnet1.lua

* Convnet from [nagadomi/kaggle-cifar10-torch7](https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/cnn_model.lua)
* Same training as mlp2
* Validation set accuracy:  0.99269
* Kaggle leaderboard on 2015-01-12: 0.97929,  25
