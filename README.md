# Datum-wise online incremental factorization for deep convolutional neural network

This repository is the official implementation of 'Datum-wise online incremental factorization for deep convolutional neural network' which is submitted to the journal 'Sensors'.

## Requirements

Python 2.7

CUDA 8.0+

CUDNN 6.0+

Caffe (PyCaffe)


To install Caffe, following link would be helpful:

- [Caffe Installation Instructions](https://caffe.berkeleyvision.org/installation.html)

When you install caffe, you should install pycaffe also. this work uses pycaffe.

To install other requirements:

```setup
pip install -r requirements.txt
```

**IMPORTANT** After installation, place this 'DOI' folder inside the caffe root directory.
The root directory name would be 'caffe' or 'caffe-master' which consists of 'src', 'tools', 'include', etc.

## Dataset

We use Cifar-100 and Cifar-10. Downloadable from this [Link](https://www.cs.toronto.edu/~kriz/cifar.html)

Please download python versions.
After download, extract inside the current 'DOI' directory.
If you have 'cifar-100-python' and 'cifar-10-batches-py' directories in the current directory, then you are ready.

Run following commands to prepare training and testing data:

```dataset
python cifar100.py
python cifar10.py
```

Check whether 'cifar-10', 'cifar-10-test', 'cifar-100', 'cifar-100-test' folders are filled with images.

## Pre-trained Models

Since this work is based on pre-trained VGGNet, you need pre-trained weight of VGGNet to run training code.
You can download pre-trained VGGNet weight here:

- [VGG pretrained on ILSVRC2012](https://drive.google.com/file/d/12nD9vJkT7u4P6SgaF1cUYSF7Y_fxMo8U/view?usp=sharing)

(Optional) You may not want to train but test directly from pretrined weights.
You can download pretrained weights here:

- [Pre-trained weights for all experiments](https://drive.google.com/file/d/1rA68TRZM0fDhemmj7W5QzQ_IOGT9Nb9E/view?usp=sharing)

## Training

To train the models in the paper, run these commands:

Class incremental learning for Cifar-100

```train1
python class_incremental.py
```

Random incremental learning for Cifar-100

```train1
python random_incremental.py
```

Class incremental learning for Cifar-10

```train1
python class_incremental_cifar10.py
```

## Evaluation

To evaluate the models in the paper, run these commands:

Class incremental learning for Cifar-100

```train1
python class_incremental_test.py
```

Random incremental learning for Cifar-100

```train1
python random_incremental_test.py
```

Class incremental learning for Cifar-10

```train1
python class_incremental_cifar10_test.py
```

Test codes request trained weights from training phase.
You should have trained weights before run test codes.

Each test code generates result 'csv' files.
If you accidentally stoped the test code, after using the 'csv' file **please remove it** before run the same test code again.
If not, the test code will write results right after the old 'csv' file.

### Detailed description of generated results

**Class incremental learning test**

It generates 'class_incremental_cifar100(or cifar10)_result.csv' file.
Final result should have 100x100(or 10x10) triangular matrix.
Each row indicates new model and each column indicates new task.
Therfore, if you want to get the average performance of each model, you should average each row.

**Random incremental learning test**

It generates 'random_incremental_result.csv' file.
Final result should have one column 100 row vector.
Each row indicates new model. The value means accuracy.

## Results

Our model achieves the following performance on :

### Cifar-100 class increment

| Model name         | Accuracy        | Forgetting     |
| ------------------ |---------------- | -------------- |
| **DOI**            |   **31.48%**    |    **34.57%**  |
| LwF                |     2.4%        |                |

### Cifar-100 random increment

| Model name         | Final Accuracy  | Maximum Accuracy |
| ------------------ |---------------- | ---------------- |
| **DOI**            |   **43.8%**     |    **45.25%**    |
| Backpropagation    |     10.43%      |      10.43%      |

### Cifar-10 class increment

| Model name         | Accuracy        | Forgetting     |
| ------------------ |---------------- | -------------- |
| GEM                |     16.8%       |      73.5%     |
| iCarl              |     28.6%       |      49%       |
| ER_MIR             |     29.8%       |      50.2%     |
| **DOI**            |   **50.4%**     |    **48.9%**   |

