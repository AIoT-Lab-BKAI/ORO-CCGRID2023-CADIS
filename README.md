# easyFL: A Lightning Framework for Federated Learning

This repository is PyTorch implementation for paper [Federated Learning with Fair Averaging](https://fanxlxmu.github.io/publication/ijcai2021/) which is accepted by IJCAI-21 Conference.

Our easyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-researchers to quickly realize and compare popular centralized federated learning algorithms. 

## Table of Contents
- [easyFL: A Lightning Framework for Federated Learning](#easyfl-a-lightning-framework-for-federated-learning)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [QuickStart](#quickstart)
  - [Citation](#citation)
  - [Contacts](#contacts)
  - [References](#references)

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
@article{wang2021federated,
  title={Federated Learning with Fair Averaging},
  author={Wang, Zheng and Fan, Xiaoliang and Qi, Jianzhong and Wen, Chenglu and Wang, Cheng and Yu, Rongshan},
  journal={arXiv preprint arXiv:2104.14937},
  year={2021}
}te
```

## Contacts
Zheng Wang, zwang@stu.xmu.edu.cn

Xiaoliang Fan, fanxiaoliang@xmu.edu.cn, https://fanxlxmu.github.io

## References
<div id='refer-anchor-1'></div>

\[McMahan. et al., 2017\] [Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.](https://arxiv.org/abs/1602.05629)

<div id='refer-anchor-2'></div>

\[Li et al., 2020\] [Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. arXiv e-prints, page arXiv:1812.06127, 2020.](https://arxiv.org/abs/1812.06127)

<div id='refer-anchor-3'></div>

\[Wang et al., 2021\] [Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang and Rongshan Yu. Federated Learning with Fair Averaging. arXiv e-prints, page arXiv:2104.14937, 2021.](https://arxiv.org/abs/2104.14937)


<div id='refer-anchor-4'></div>

\[Li et al., 2019\] [ Tian Li, Maziar Sanjabi, and Virginia Smith. Fair resource allocation in federated learning. CoRR, abs/1905.10497, 2019.](https://arxiv.org/abs/1905.10497)

<div id='refer-anchor-5'></div>

\[Mohri et al., 2019\] [Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh. Agnostic federated learning. CoRR, abs/1902.00146, 2019.](https://arxiv.org/abs/1902.00146)

<div id='refer-anchor-6'></div>

\[Hu et al., 2020\] [Zeou Hu, Kiarash Shaloudegi, Guojun Zhang, and Yaoliang Yu. Fedmgda+: Federated learning meets multi-objective optimization. arXiv e-prints, page arXiv:2006.11489, 2020.](https://arxiv.org/abs/2006.11489)

<div id='refer-anchor-7'></div>

\[Huang et al., 2020\] [Wei Huang, Tianrui Li, Dexian Wang, Shengdong Du, and Junbo Zhang. Fairness and accuracy in federated learning. arXiv e-prints, page arXiv:2012.10069, 2020.](https://arxiv.org/abs/2012.10069) 

<div id='refer-anchor-8'></div>

\[Li et al., 2021\][Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng. Federated Learning on Non-IID Data Silos: An Experimental Study. arXiv preprint arXiv:2102.02079, 2021.](https://arxiv.org/abs/2102.02079)

<div id='refer-anchor-9'></div>

\[Caldas et al., 2018\] [Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, Ameet Talwalkar. LEAF: A Benchmark for Federated Settings. arXiv preprint arXiv:1812.01097, 2018.](https://arxiv.org/abs/1812.01097)

<div id='refer-anchor-10'></div>

\[He et al., 2020\] [He, Chaoyang and Li, Songze and So, Jinhyun and Zhang, Mi and Wang, Hongyi and Wang, Xiaoyang and Vepakomma, Praneeth and Singh, Abhishek and Qiu, Hang and Shen, Li and Zhao, Peilin and Kang, Yan and Liu, Yang and Raskar, Ramesh and Yang, Qiang and Annavaram, Murali and Avestimehr, Salman. FedML: A Research Library and Benchmark for Federated Machine Learning. arXiv preprint arXiv:2007.13518, 2020.](https://arxiv.org/abs/2007.13518)

<div id='refer-anchor-11'></div>

\[Karimireddy et al., 2020\] [Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh, SCAFFOLD: Stochastic Controlled Averaging for Federated Learning, Proceedings of the 37th International Conference on Machine Learning, PMLR 119:5132-5143, 2020.](https://arxiv.org/abs/1910.06378v3)

<div id='refer-anchor-12'></div>

[Acar et al., 2021] [Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, Venkatesh Saligrama. Federated Learning Based on Dynamic Regularization. International Conference on Learning Representations (ICLR), 2021](https://openreview.net/forum?id=B7v4QMR6Z9w)

