# easyFL: A Lightning Framework for Federated Learning

This repository is PyTorch implementation for paper **CADIS: Handling Cluster-skewed Non-IID Data in Federated Learning with Clustered Aggregation and Knowledge DIStilled Regularization** which is accepted by CCGRID-23 Conference.

The repository is a modified version of [easyFL](https://github.com/WwZzz/easyFL), which was introduced by the authors in **Federated Learning with Fair Averaging** (IJCAI-21). EasyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-researchers to quickly realize and compare popular centralized federated learning algorithms. 

## Requirements

The project is implemented using Python3 with dependencies below:

```
numpy>=1.17.2
pytorch>=1.3.1
torchvision>=0.4.2
cvxopt>=1.2.0
scipy>=1.3.1
matplotlib>=3.1.1
prettytable>=2.1.0
ujson>=4.0.2
wandb>=0.13.5
```

## QuickStart

To run the experiments, simply:
'''bash

CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm cadis --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/quantitative/MNIST-noniid-quantitative_1.json --num_rounds 100 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

'''

**Note**: If you do not have a wandb account, you can set "wandb 0" as an argument in bash command.

Run the bash files to reproduce the results.

The results will be stored in the folder: "./fedtask" which will be automatically created when running the bash file.

To visualize the results, run the result_analysis.py in the folder: "./utils". Make sure you give the correct path.

## Citation

Please cite our paper in your publications if this code helps your research.

```
@article{hung2023cadis,
  title={CADIS: Handling Cluster-skewed Non-IID Data in Federated Learning with Clustered Aggregation and Knowledge DIStilled Regularization},
  author={Nang Hung Nguyen, Duc Long Nguyen, Trong Bang Nguyen, Thanh-Hung Nguyen, Hieu Pham, Truong Thao Nguyen and Phi Le Nguyenn},
  booktitle={The 23rd International Symposium on Cluster, Cloud and Internet Computing},
  year={2023}
}
```
