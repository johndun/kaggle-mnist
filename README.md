# Kaggle-MNIST

### mlp1.lua

* Single hidden layer: 576 -> 2048 -> softmax
* No data augmentation
* Single learning rate: 1.0
* Momentum: 0.9, decay: 1e-6 (`learning_rate = lr / (1 + n_batches * 1e-6)`)
* Validation set accuracy: 0.97822
* Kaggle leaderboard on 2015-01-11 (accuracy, rank): 0.97957,  127

### mlp2.lua

* Same model as mlp1.lua
* No data augmentation
* 2 learning rates: 1.0 and 0.1 (the number of 
  training batches at each selected using the validation set)
* Momentum: 0.9, decay: 1e-6
* Validation set accuracy: 0.97872
* Kaggle leaderboard on 2015-01-12: 0.97929, 127


### convnet1.lua

* Convnet from [nagadomi/kaggle-cifar10-torch7](https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/cnn_model.lua)
* Same training as mlp2
* Validation set accuracy:  0.99269
* Kaggle leaderboard on 2015-01-12: 0.97929,  25
