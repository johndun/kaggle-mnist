# Kaggle-MNIST

### mlp1.lua

* Single hidden layer: 576 -> 2048 -> softmax
* No data augmentation
* 2 learning rates: 1.0 and 0.1 (the number of 
  training batches at each selected using the validation set)
* Momentum: 0.9, decay: 1e-6 `learning_rate = lr / (1 + n_batches * 1e-6)`
* Validation set accuracy: ??

### convnet1.lua

* Convnet from [nagadomi/kaggle-cifar10-torch7](https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/cnn_model.lua)
* No data augmentation
* Same training as mlp1
* Validation set accuracy: ??

### convnet2.lua

* Convnet from [nagadomi/kaggle-cifar10-torch7](https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/cnn_model.lua)
* Data augmentation during training (random cropping and zooming) and testing (3x3 grid, 2 scales)
* Same training as mlp1
* Validation set accuracy: 99.59

### convnet3.lua

* VGG-style model
* Data augmentation as convnet2
* Same training as mlp1
* Validation set accuracy: 99.51
