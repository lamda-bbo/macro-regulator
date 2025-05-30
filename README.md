# NeurIPS'24 Reinforcement Learning Policy as Macro Regulator Rather than Macro Placer

Official implementation of NeurIPS'24 paper "Reinforcement Learning Policy as Macro Regulator Rather than Macro Placer"

This repository contains the Python code for MaskRegulate, a reinforcement learning implementation for macro placement. Formulated as a regulator rather than a placer and equipped with RegularMask, MaskRegulate empirically achieves significant improvements over previous methods.

## Requirements
+ python==3.8.5
+ torch==1.7.1
+ torchvision==0.8.2
+ torchaudio==0.7.2
+ pyyaml==5.3.1
+ gym==0.22.0
+ Shapely==2.0.4
+ matplotlib==3.4.3
+ cairocffi==1.7.0
+ tqdm==4.61.2
+ tensorboard==2.14.0 
+ scikit_learn==1.3.2
+ numpy==1.21.2

## File structure

+ `benchmark` directory stores the benchmarks for running. Please download the ICCAD2015 benchmark and move it to `benchmark/` (i.e., `benchmark/superblue1`).
+ `config` stores the hyperparameters for our algorithm.
+ `DREAMPlace_source` serves as a thirdparty standard cell placer borrowed from [DREAMPlace](<https://github.com/limbo018/DREAMPlace>).
+ `policy` stores a pretrained policy trained on `superblue1`, `superblue3`, `superblue4` and `superblue5`.
+ `src` contains the source code of MaskRegulate.
+ `utils` defines some functions to be used for optimization.
  
## Usage
Please first download the docker image from [Baidu Netdisk](https://pan.baidu.com/s/1GAu1-RVA5IYHd1LjyL2Xww?pwd=syur) or [DREAMPlace](<https://github.com/limbo018/DREAMPlace>), and compile `DREAMPlace_source` in the docker container following the below commands:
```
cd DREAMPlace_source
mkdir build
cmake .. -DCMAKE_INSTALL_PREFIX=../../DREAMPlace
make
make install
```

After that, please download the ICCAD2015 benchmark via [Google Drive](https://drive.google.com/file/d/1JEC17FmL2cM8BEAewENvRyG6aWxH53mX/view?usp=sharing).

### Parameters
+ `--seed` random seed for running.
+ `--gpu` GPU ID used by the algorithm execution.
+ `--episode` number of episodes for training.
+ `--checkpoint_path` the saved model to be loaded.
+ `--eval_policy` only evalute the policy given by `--checkpoint_path`.
+ `--dataset_path` the placement file to regulate. MaskRegulate will improve the chip layout obtained from [DREAMPlace](<https://github.com/limbo018/DREAMPlace>) if `--dataset_path` is not provided. Currently, MaskRegulate only supports training on a single benchmark when a `--dataset_path` is provided (i.e., if `superblue1_reference.def` is given, please set `--benchmark_train=[superblue1]` and `--dataset_path=./superblue1_reference.def`).

### Run a training task
Please first navigate to the `src` directory.
```
python main.py --benchmark_train=[Benchmark1,Benchmark2] --benchmark_eval=[Benchmark1',Benchmark2']
```
- `--benchmark_train`  contains the benchmarks to train on.
- `--benchmark_eval` contains the benchmarks to evaluate on.

For example, if you want to train MaskRegulate on benchmark `superblue1`, `superblue3` and evaluate the performance on `superblue1`, `superblue3`, `superblue5`, run our command as shown bellow:
```
python main.py --benchmark_train=[superblue1,superblue3] --benchmark_eval=[superblue1,superblue3,superblue5]
```
Script `run_train.sh` is provided for a quick start.

### Run a testing task
We also provide a pre-trained model trained on `superblue1`, `superblue3`, `superblue4` and `superblue5` in `policy/pretrained_model.pkl`, which can be loaded and evaluated. For example, run the following command to test our policy on superblue1:
```
python main.py --benchmark_train=[] --benchmark_eval=[superblue1] --check_point_path=../policy/pretrained_model.pkl --eval_policy=True
```
Script `run_test.sh` is provided for a quick start.


## Citation
```
@inproceedings{macro-regulator,
    author = {Ke Xue, Ruo-Tong Chen, Xi Lin, Yunqi Shi, Shixiong Kai, Siyuan Xu, Chao Qian.},
    title = {Reinforcement Learning Policy as Macro Regulator Rather than Macro Placer},
    booktitle = {Advances in Neural Information Processing Systems 38 (NeurIPS’24)},
    year = {2024},
    pages = {140565--140588},
    address={Vancouver, Canada}
}
```

