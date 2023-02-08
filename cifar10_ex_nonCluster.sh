mkdir fedtask

# cifar10_resnet9_featured_N10_K10 - featured
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_featured_N10_K10 --model resnet9 --wandb 0 --algorithm cadis --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_featured.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_featured_N10_K10 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_featured.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_featured_N10_K10 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_featured.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_featured_N10_K10 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_featured.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_featured_N10_K10 --model resnet9 --algorithm fedfa --alpha 0.5 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_featured.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0

# cifar10_resnet9_pareto_N10_K10 - pareto
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_pareto_N10_K10 --model resnet9 --algorithm cadis --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_pareto.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_pareto_N10_K10 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_pareto.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_pareto_N10_K10 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_pareto.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_pareto_N10_K10 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_pareto.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_pareto_N10_K10 --model resnet9 --algorithm fedfa --alpha 0.5 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_pareto.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0

# cifar10_resnet9_quantitative_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N10_K10 --model resnet9 --algorithm cadis --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_quantitative.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N10_K10 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_quantitative.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N10_K10 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_quantitative.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N10_K10 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_quantitative.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N10_K10 --model resnet9 --algorithm fedfa --alpha 0.5 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/cluster/10client/CIFAR10_10client_quantitative.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
