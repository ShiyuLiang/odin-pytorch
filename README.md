# Cordial Perturbations: Detecting Out-of-Distribution Examples in Neural Networks 


This is a [PyTorch](http://pytorch.org) implementation of the detection of out-of-distribution examples in neural networks as described in the paper [Principled Detection of Out-of-Distribution Examples in Neural Networks]() by S. Liang, Y. Li and R. Srikant. 

## Expereimental Results

We used two neural network architectures, [DenseNet-BC](https://arxiv.org/abs/1608.06993) and [Wide ResNet](https://arxiv.org/abs/1605.07146).
The PyTorch implementation of [DenseNet-BC](https://arxiv.org/abs/1608.06993) is provided [here](https://github.com/andreasveit/densenet-pytorch) by Andreas Veit and [here](https://github.com/bamos/densenet.pytorch) by Brandon Amos. The PyTorch implementation of [Wide ResNet](https://arxiv.org/abs/1605.07146) is provided  [here](https://github.com/szagoruyko/wide-residual-networks) by Sergey Zagoruyko.
The experimental results are shown as follows. The definition of each metric can be found in the [paper]().
![performance](./figures/performance.png)

 



## Pre-trained Models

We provide four pre-trained neural networks: (1) two [DenseNet-BC](https://arxiv.org/abs/1608.06993) networks trained on  CIFAR-10 and CIFAR-100 respectively; (2) two [Wide ResNet](https://arxiv.org/abs/1605.07146) networks trained on CIFAR-10 and CIFAR-100 respectively. The test error rates is provided by the following table:

Architecture    |  CIFAR-10   | CIFAR-100
------------    |  ---------  | ---------
DenseNet-BC     |  4.81       | 22.37
Wide ResNet     |  3.71       | 19.86


## Running the code (DenseNet-BC)

### Dependencies

* CUDA 8.0
* PyTorch
* Anaconda2 or 3
* At least **one** GPU

### Downloading  Out-of-Distribtion Datasets
We provide download links of five out-of-distributin datasets:

* **Tiny-ImageNet (crop)**: [https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)
* **Tiny-ImageNet (resize)**:[https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* **LSUN (crop)**: [https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
* **LSUN (resize)**: [https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* **iSUN**: [https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

Here is an example code of downloading Tiny-ImageNet (crop) dataset. In the **root** directory, run

```
mkdir data
cd data
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet.tar.gz
tar -xvzf Imagenet.tar.gz
cd ..
```

### Downloading Neural Network Models

We provide download links of two DenseNet-BC networks. 

* **DenseNet-BC trained on CIFAR-10**: [https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz](https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz)
* **DenseNet-BC trained on CIFAR-100**: [https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz](https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz)

Here is an example code of downloading DenseNet-BC trained on CIFAR-10. In the **root** directory, run

```
mkdir models
cd models
wget https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz
tar -xvzf densenet10.pth.tar.gz
cd ..
```


### Running

Here is an example code reproducing the results of DenseNet-BC trained on CIFAR-10 where TinyImageNet (crop) is the out-of-distribution datasets, temperature is 1000 and perturbation magnitude is 0.0014. In the **root** directory, run

```
cd code
# model: DenseNet-BC, in-distribution: CIFAR-10, out-distribution: TinyImageNet (crop)
# magnitude: 0.0014, temperature 1000, gpu: 0
python main.py --nn densenet10 --out_dataset Imagenet --magnitude 0.0014 --temperature 1000 --gpu 0
```
**Note:** Please correctly choose arguments in the following way. 

#### args
* **args.nn**: the arguments of neural networks are shown as follows
	
	Nerual Network Models | args.nn
	----------------------|--------
	DenseNet-BC trained on CIFAR-10| densenet10
	DenseNet-BC trained on CIFAR-100| densenet100
* **args.out_dataset**: the arguments of out-of-distribution datasets are shown as follows

	Out-of-Distribution Datasets     | args.out_dataset
	------------------------------------|-----------------
	Tiny-ImageNet (crop)                | Imagenet
	Tiny-ImageNet (resize)              | Imanenet_resize
	LSUN (crop)                         | LSUN
	LSUN (resize)                       | LSUN_resize
	iSUN                                | iSUN
	Uniform random noise                | Uniform
	Gaussian random noise               | Gaussian

* **args.magnitude**: the arguments of noise magnitude are shown as follows

	Out-of-Distribution Datasets        |   args.magnitude (densenet10)     |  args.magnitude (densenet100)
	------------------------------------|------------------|-------------
	Tiny-ImageNet (crop)                | 0.0014           | 0.0014        
	Tiny-ImageNet (resize)              | 0.0014           | 0.0028
	LSUN (crop)                         | 0                | 0.0028
	LSUN (resize)                       | 0.0014           | 0.0028
	iSUN                                | 0.0014           | 0.0028
	Uniform random noise                | 0.0014           | 0.0028
	Gaussian random noise               | 0.0014           |0.0028

* **args.temperature**: temperature is set to 1000 in all cases. 
* **args.gpu**: use gpu0 to run the code.

### Outputs
Here is an example of output. 

```
Neural network architecture:          DenseNet-BC-100
In-distribution dataset:                     CIFAR-10
Out-of-distribution dataset:     Tiny-ImageNet (crop)

                          Baseline         Our Method
FPR at TPR 95%:              34.8%               4.3% 
Detection error:              9.9%               4.6%
AUROC:                       95.3%              99.1%
AUPR In:                     96.4%              99.2%
AUPR Out:                    93.8%              99.1%
```

## Running the code (Wide ResNet)

### Dependencies

* CUDA 8.0
* PyTorch
* Anaconda2 or 3
* At least **three** GPUs

**Note:** Now we require three gpus for reproducing the results for Wide ResNet. Single gpu version is coming.

### Downloading Out-of-Distribution Datasets
The codes for downloading the out-of-distribution datasets are exactly the same as the code shown in the previous section.  

* **Tiny-ImageNet (crop)**: [https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)
* **Tiny-ImageNet (resize)**: [https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* **LSUN (crop)**: [https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
* **LSUN (resize)**: [https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
* **iSUN**: [https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

Here is an example code of downloading LSUN (crop) dataset. In the **root** directory, run

```
mkdir data
cd data
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf Imagenet.tar.gz
```

### Downloading Neural Network Models

We provide download links of two Wide ResNet networks. 

* **Wide ResNet trained on CIFAR-10**: [https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz](https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz)
* **Wide ResNet trained on CIFAR-100**: [https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet100.pth.tar.gz](https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet100.pth.tar.gz)

Here is an example code of downloading Wide ResNet trained on CIFAR-10. In the **root** directory, run

```
mkdir models
cd models
wget https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz
tar -xvzf wideresnet10.pth.tar.gz
```


### Running

Here is an example code reproducing the results of Wide ResNet trained on CIFAR-10 where LSUN (crop) is the out-of-distribution datasets, temperature is 1000 and perturbation magnitude is 0. In the **root** directory, run

```
cd code
# model: Wide ResNet, in-distribution: CIFAR-10, out-distribution: LSUN (crop)
# magnitude: 0.0014, temperature 1000
# gpu: 1 for wideresnet10, 2 for wideresnet100
python main.py --nn wideresnet10 --out_dataset LSUN --magnitude 0 --temperature 1000 --gpu 1
```
**Note:** Please correctly choose arguments in the following way. 

#### args
* **args.nn**: the arguments of neural networks are shown as follows
	
	Nerual Network Models | args.nn
	----------------------|--------
	DenseNet-BC trained on CIFAR-10| wideresnet10
	DenseNet-BC trained on CIFAR-100| wideresnet100
	
* **args.out_dataset**: the arguments of out-of-distribution datasets are shown as follows

	Out-of-Distribution Datasets     | args.out_dataset
	------------------------------------|-----------------
	Tiny-ImageNet (crop)                | Imagenet
	Tiny-ImageNet (resize)              | Imanenet_resize
	LSUN (crop)                         | LSUN
	LSUN (resize)                       | LSUN_resize
	iSUN                                | iSUN
	Uniform random noise                | Uniform
	Gaussian random noise               | Gaussian

* **args.magnitude**: the arguments of noise magnitude are shown as follows

	Out-of-Distribution Datasets        |   args.magnitude (wideresnet10)     |  args.magnitude (wideresnet100)
	------------------------------------|------------------|-------------
	Tiny-ImageNet (crop)                | 0.0005           | 0.0028        
	Tiny-ImageNet (resize)              | 0.0011           | 0.0028
	LSUN (crop)                         | 0                | 0.0048
	LSUN (resize)                       | 0.0006           | 0.002
	iSUN                                | 0.0008           | 0.0028
	Uniform random noise                | 0.0014           | 0.0028
	Gaussian random noise               | 0.0014           | 0.0028

* **args.temperature**: temperature is set to 1000 in all cases. 
* **args.gpu**: please use the following gpu to run the code:
	
	Neural Network Models |  args.gpu
	----------------------|----------
	wideresnet10          | 1
	wideresnet100         | 2

	**Note:** Here we require gpu1 and gpu2 for reproducing the results of Wide ResNet. The single gpu (gpu0) version will be released in the future. 

### Outputs
Here is an example of output. 

```
Neural network architecture:        Wide-ResNet-28-10
In-distribution dataset:                     CIFAR-10
Out-of-distribution dataset:              LSUN (crop)

                          Baseline         Our Method
FPR at TPR 95%:              35.3%              22.0% 
Detection error:             10.8%               9.8%
AUROC:                       94.4%              95.8%
AUPR In:                     95.0%              95.8%
AUPR Out:                    93.0%              95.4%
```



