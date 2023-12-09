# TransHP
[NeurIPS 2023] The official implementation of [TransHP: Image Classification with Hierarchical Prompting](https://arxiv.org/pdf/2304.06385.pdf).

![image](https://github.com/WangWenhao0716/TransHP/blob/main/demo.png)

## Environment
Our TransHP uses PyTorch 1.8.0 and timm 0.4.12. They can be easily installed by:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
```
pip install timm==0.4.12
```

The minimum hardware requirement of our TransHP is 8 V100. 

## Dataset
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision ```datasets.ImageFolder```, and the training and validation data is expected to be in the ```train/``` folder and ```val/``` folder respectively.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```


## Train
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
--model deit_small_hi_patch16_224 --batch-size 128 --data-path /path/to/imagenet/ \
--output_dir ./ckpt/TransHP/
```

## Test

We release our [trained model](https://drive.google.com/file/d/1FkPhq8Y2Lcgv-JZ2SM8wcV2voXxfr9UJ/view?usp=sharing) and corresponding [logs](https://drive.google.com/file/d/1YaMP4RDNZonqREthpQ4OnHAXccR0rKb1/view?usp=sharing), you should download and save it. Then you can test its performance by
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --eval \
--resume released_checkpoint.pth --model deit_small_hi_patch16_224 \
--data-path /path/to/imagenet/
```

This should give
```
* Acc@1 78.514 Acc@5 93.614 loss 0.948
```

*Note that this is a bit different from our reported performance (78.65) in our paper due to the randomness in the reproducing.*

## Known issues
1. If the loss is NaN, please check https://github.com/facebookresearch/deit/issues/29.

2. One limitation of our TransHP is its need to select prompting blocks and tune the balance parameters. Any advice or follow-up work to solve this problem is welcomed.

3. Another limitation is most of our experiments are performed on a lightweight Vision Transformer (ViT) due to insufficient computing resources. I am happy to see more experiments on larger ViTs or bigger datasets.

## Citation
```
@inproceedings{
    wang2023transhp,
    title={Trans{HP}: Image Classification with Hierarchical Prompting},
    author={Wenhao Wang and Yifan Sun and Wei Li and Yi Yang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=vpQuCsZXz2}
}
```
## Acknowledgement
We implement our TransHP based on [DeiT](https://github.com/facebookresearch/deit/blob/main/README_deit.md). Our baseline is a lightweight ViT, i.e. ViT-small.


