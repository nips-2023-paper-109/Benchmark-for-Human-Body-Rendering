# A Comprehensive Benchmark for Neural Human Radiance Fields

## Installation

We recommend to use [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda create -n humanbench
    conda activate humanbench
    pip install -r requirements.txt

## Prepare Datasets
Download corresponding datasets [ZJU-MoCap](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset), [GeneBody](https://generalizable-neural-performer.github.io/genebody.html)
and [HuMMan](https://caizhongang.github.io/projects/HuMMan/recon.html), and put them to the correct path according to the README.md files in each method.

### `Download SMPL model`

Download the gender neutral SMPL model from [here](https://smplify.is.tue.mpg.de/), and unpack **mpips_smplify_public_v2.zip**.

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.


## Unified Evaluation
For unified evaluation, we put all the official implementations of each method together, and modified their evaluation settings (human cropping, test views, test frames and etc.) to be the same.

## Generalization
For scene-specific methods, we need to use part of GeneBody and HuMMan datasets. The names of the used subjects are shown as follows. For GeneBody dataset, the official provided SMPL parameters are for SMPL-X, so we use [this](https://github.com/vchoutas/smplx) tool to convert smpl-x parameters to smpl parameters.

GeneBody
- Normal: zhanghongwei, zhuxuezhi, zhanghao
- Hard Clothing: jinyutong2, fuzhizhi2, huajiangtao3
- Hard Pose: zhuna2, ivan, dilshod

HuMMan:
- Male: p000447_a000221, p000455_a000986, p000457_a000074, p000484_a000285, p100047_a001425
- Female: p000452_a000208, p000480_a000986, p000498_a000221, p000471_a000213, p000475_a000063

After preparing the each subjects, run [NeuralBody](https://github.com/zju3dv/neuralbody) and [HumanNeRF](https://github.com/chungyiweng/humannerf) according to their official instructions.


To train neuralbody on a subject (e.g. zhanghao) of GeneBody, run
```
python train_net.py --cfg_file configs/genebody/latent_xyzc_zhanghao.yaml exp_name genebody_zhanghao resume False
```

To train neuralbody on a subject of HuMMan is similar.

To evaluate neuralbody on a subject (e.g. zhanghao) of GeneBody, run
```
python run.py --type evaluate --cfg_file configs/genebody/latent_xyzc_zhanghao.yaml exp_name genebody_zhanghao
```
To evaluate neuralbody on a subject of HuMMan is similar.

To train HumanNeRF on a subject (e.g. zhanghao) of GeneBody, preprocess the dataset first, i.e.
```
cd tools\prepare_genebody
python prepare_dataset.py --cfg zhanghao.yaml
```
Then, 
```
cd ..
python train.py --cfg configs/human_nerf/genebody/zhanghao/adventure.yaml
```
To train HumanNeRF on a subject of HuMMan is similar.

To evaluate HumanNeRF on a subject (e.g. zhanghao) of GeneBody, preprocess the dataset first, i.e.
```
cd tools\prepare_genebody
python prepare_dataset_eval.py --cfg zhanghao_eval.yaml
```
Then,
```
cd ..
python eval.py --cfg configs/human_nerf/genebody/zhanghao/adventure.yaml
```

For generalizable methods, to train GP-NeRF on HuMMan, run
```
python tools/train.py --cfg configs/trainhumman_valhumman.yaml
```

To evaluate GP-NeRF on HuMMan-eval, run
```
python3 tools/inference.py --cfg configs/trainhumman_valhumman.yaml render.resume_path logs/cam3humman_hummanval/cam3humman_hummanval/xxx.pth test.test_seq 'demo_trainhumman_valhumman'  test.is_vis True dataset.test.sampler 'FrameSampler' dataset.test.shuffle False render.file 'demo_render'
```

## Animation
To train our GeneHumanNeRF on HuMMan dataset, run
```
python train_net.py --cfg_file configs/humman/humman.yaml exp_name humman_full_train resume False
```

To evaluate on a subject (e.g. p000455_a000986) of HuMMan-eval dataset, run
```
python run.py --type evaluate --cfg_file configs/humman_eval/p000455_a000986.yaml exp_name humman_full_train resume True
```


