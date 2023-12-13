# Out-of-Distribution Detection in Long-Tailed Recognition with Calibrated Outlier Class Learning

This is the official implementation of the [Out-of-Distribution Detection in Long-Tailed Recognition with Calibrated Outlier Class Learning]()

## Dataset Preparation

### In-distribution dataset

Please download [CIFAR10](), [CIFAR100](), and [ImageNet-LT](https://liuziwei7.github.io/projects/LongTail.html) , place them  in`./datasets` 

### Auxiliary/Out-of-distribution dataset

For [CIFAR10-LT]() and [CIFAR100-LT](), please download [TinyImages 300K Random Images]() for auxiliary in `./datasets` 

For [CIFAR10-LT]() and [CIFAR100-LT](), please download [SC-OOD](https://jingkang50.github.io/projects/scood) benchmark  for out-of-distribution in `./datasets` 

For [ImageNet-LT]([Large-Scale Long-Tailed Recognition in an Open World (liuziwei7.github.io)](https://liuziwei7.github.io/projects/LongTail.html)), please download [ImageNet10k_eccv2010](https://image-net.org/data/imagenet10k_eccv2010.tar) benchmark for auxiliary and out-of-distribution in `./datasets` 

All datasets follow [PASCL](https://github.com/amazon-science/long-tailed-ood-detection)

## Training

### CIFAR10-LT: 

```
python train.py --gpu 0 --ds cifar10 --Lambda1 0.05 --Lambda2 0.05 --Lambda3 0.1 --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

### CIFAR100-LT:

```
python train.py --gpu 0 --ds cifar100 --Lambda1 0.05 --Lambda2 0.05 --Lambda3 0.1  --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

### ImageNet-LT:

```
python stage1.py --gpu 0,1,2,3 --ds imagenet --md ResNet50 --lr 0.1 --Lambda1 0.02 --Lambda2 0.01 --Lambda3 0.01 --drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

## Testing

### CIFAR10-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar10 --dout $dout \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
done
```

### CIFAR100-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar100 --dout $dout \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
done
```

### ImageNet-LT:

```
python test_imagenet.py --gpu 0 \
    --drp <where_you_store_all_your_datasets> \
    --ckpt_path <where_you_save_the_ckpt>
```


## Acknowledgment

Part of our codes are adapted from these repos:

Outlier-Exposure - https://github.com/hendrycks/outlier-exposure - Apache-2.0 license

PASCL - https://github.com/amazon-science/long-tailed-ood-detection - Apache-2.0 license

Open-Sampling - https://github.com/hongxin001/open-sampling - Apache-2.0 license

Long-Tailed-Recognition.pytorch - https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch - GPL-3.0 license

## License

This project is licensed under the Apache-2.0 License.
