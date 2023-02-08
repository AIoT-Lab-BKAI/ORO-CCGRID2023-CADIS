mkdir fedtask

# mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/quantitative/MNIST-noniid-quantitative_1.json --num_rounds 100 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm fedfa --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm feddyn --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm cadis --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/quantitative/MNIST-noniid-quantitative_1.json --num_rounds 100 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 50 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 50 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm feddyn --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 50 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm fedfa --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 50 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm cadis --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 50 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K20 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K20 --model cnn --algorithm cadis --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K20 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K20 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K20 --model cnn --algorithm feddyn --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K20 --model cnn --algorithm fedfa --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K40 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K40 --model cnn --algorithm fedfa --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K40 --model cnn --algorithm feddyn --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K40 --model cnn --algorithm cadis --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K40 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K40 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K80 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K80 --model cnn --algorithm feddyn --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K80 --model cnn --algorithm fedfa --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K80 --model cnn --algorithm cadis --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K80 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K80 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/mnist/data --log_folder fedtask --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
