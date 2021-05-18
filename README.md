# <img src="docs/source/images/logo/libkge-header-2880.png" alt="Distributed LibKGE: A knowledge graph embedding library" width="80%">

**NOTE: The preparation of this repository is still ongoing**


This is the code and configuration accompanying the paper "Parallel Training of Knowledge Graph Embedding Models: A Comparison of Techniques".
The code extends the knowledge graph embedding library [LibKGE](https://github.com/uma-pi1/kge).
For documentation on LibKGE refer to LibKGE repository.
We provide the hyper-parameter settings for the experiments in their corresponding configuration files.

## Quick start

```sh
# retrieve and install project in development mode
git clone https://github.com/AdrianKs/dist-kge.git
cd dist-kge
pip install -e .

# download and preprocess datasets
cd data
sh download_all.sh
cd ..

# train an example model on toy dataset (you can omit '--job.device cpu' when you have a gpu)
kge start examples/toy-complex-train.yaml --job.device cpu

```
This example will train on a toy dataset in a sequential setup on CPU


## Dataset preparation for parallel training
**NOTE: Freebase already comes with multiple partition settings to save preprocessing time**

To partition the data run the following commands (you only need to do this once)

**Random Partitioning**

For random partitioning no further preparation is needed.

**Relation Partitioning**
```sh
cd data
python partition_relation.py <dataset-name> <num-partitions>
cd ..
```

**Stratification**
```sh
cd data 
python partition_stratification.py <dataset-name> <num-partitions>
cd ..
```

**Graph-Cut**

````sh
cd data
python partition_graph_cut.py <dataset-name> <num-partitions>
cd ..
````


### Single Machine Multi-GPU Training
Run following example to train on two GPUs with random partitioning (two worker per GPU):
````
python -m kge start examples/fb15k-complex-parallel.yaml
````
The most important configuration options for multi-gpu training are:
````yaml
import:
  - complex
  - distributed_model
model: distributed_model
distributed_model:
  base_model: complex
job:
  distributed:
    num_partitions: 4
    num_workers: 4
    partition_type: random
    master_port: 8888  # change in case this port is used on your machine
  device_pool:
    - cuda:0
    - cuda:1
train:
  type: distributed_negative_sampling
  optimizer:
    default:
      type: dist_adagrad
````

### Multi-GPU Multi-Machine Training
#### Parameter Server
For multi-machine training we rely on the parameter server [Lapse](https://github.com/alexrenz/lapse-ps).
To install Lapse and the corresponding python bindings run the following commands:
````sh
git clone https://github.com/alexrenz/lapse-ps.git
cd lapse-ps
make ps KEY_TYPE=int64_t CXX11_ABI=$(python bindings/lookup_torch_abi.py) DEPS_PATH=$(pwd)/deps_bindings
cd bindings 
python setup.py install --user
````
For further documentation on the python bindings refer to [Lapse-Binding documentation](https://github.com/alexrenz/lapse-ps/tree/main/bindings).

In case you can not use Lapse, we provide a very inefficient parameter server (for debugging). To use this debugging PS use the option `--job.distributed.parameter_server torch`

#### Interface
As we use the gloo backend to communicate between master and worker nodes you need to specify the interface connecting your machines and specify it as `--job.distributed.gloo_socket_ifname`.
You can find out the names of your interfaces with the command
````sh
ip address
````

#### Example
Run the following example to train on two machines with one GPU each (1@2) with random partitioning:

Command for machine 1
````sh
python -m kge start examples/fb15k_complex_parallel.yaml --job.distributed.machine_id 0 --job.distributed.master_ip <some_ip>
````

Command for machine 2
````sh
python -m kge start examples/fb15k_complex_parallel.yaml --job.distributed.machine_id 1 --job.distributed.master_ip <some_ip>
````


Important options for distributed training in addition to the options specified in the single-machine setting are:
````yaml
job:
  distributed:
    master_ip: <some ip>  # ip address of one of your machines
    num_machines: 2
    num_workers_machine: 2
    gloo_socket_ifname: "bond0"  # name of the interface to use. Use command 'ip address' to find names
    parameter_server: "lapse"
````


## Experiments and Configuration
### Sequential training
The best hyper-parameter setting per dataset and model are

**ComplEx**

partition scheme    |   epoch time  |   time to 95% MRR |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   ------  |
FB15k   |   5.7s    |   2.4min  |   0.804   |   [config](examples/experiments/fb15k/complex-fb15k-sequential.yaml)
Yago3-10    |   24.3s   |   38.5min |   0.542   |   [config](examples/experiments/yago3-10/complex-yago3-10-sequential.yaml)
Wikidata    |   438.4s  |   219.0min    |   0.297   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-sequential.yaml)
Freebase (main memory)    | 6455.5s |   -   |   0.344   |   [config](examples/experiments/freebase/complex-freebase-sequential.yaml)


**RotatE**

partition scheme    |   epoch time  |   time to 95% MRR |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   ------  |
FB15k   |   15.9s   |   14.5min |   0.780   |   [config](examples/experiments/fb15k/rotate-fb15k-sequential.yaml)
Yago3-10    |   74.1s   |   259.3s  |   0.451   |   [config](examples/experiments/yago3-10/rotate-yago3-10-sequential.yaml)
Wikidata    |   798.4s  |   199.6min    |   0.258   |   [config](examples/experiments/wikidata5m/rotate-wikidata5m-sequential.yaml)
Freebase (main memory)    | 7785.4s |   -   |   0.571   |   [config](examples/experiments/freebase/rotate-freebase-sequential.yaml)

### Multi-GPU, Multi-Machine Training
#### FB15k

**ComplEx**

partition scheme    |   epoch time (1@2)  |   time to 95% MRR (1@2) |   data sent   |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
random  |   4.0s   |   **1.7min**  |   2.0GB   |    **0.800**   |   [2@1](examples/experiments/fb15k/complex-fb15k-parallel-random.yaml),[1@2](examples/experiments/fb15k/complex-fb15k-distributed-random.yaml)
relation  |   **3.6s** |   2.1min  |   1.7GB   |   **0.800**   |   [2@1](examples/experiments/fb15k/complex-fb15k-parallel-relation.yaml),[1@2](examples/experiments/fb15k/complex-fb15k-distributed-relation.yaml)
stratification (CAR)  |   6.9s |   8.6min  |   **0.7GB**   |   0.799   |   [2@1](examples/experiments/fb15k/complex-fb15k-parallel-stratification-car.yaml),[1@2](examples/experiments/fb15k/complex-fb15k-distributed-stratification-car.yaml)
graph-cut  |   3.9s    |   -   |   0.9GB   |   0.601   |   [2@1](examples/experiments/fb15k/complex-fb15k-parallel-graph-cut.yaml),[1@2](examples/experiments/fb15k/complex-fb15k-distributed-graph-cut.yaml)

**RotatE**

partition scheme    |   epoch time (1@2)  |   time to 95% MRR (1@2) |   data sent   |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
random  |   9.5s   |   11.0min |   2.2GB   |   0.774   |   [2@1](examples/experiments/fb15k/rotate-fb15k-parallel-random.yaml),[1@2](examples/experiments/fb15k/rotate-fb15k-distributed-random.yaml)
relation  |   **8.9s**  |   12.6min |   1.8GB   |   0.774   |   [2@1](examples/experiments/fb15k/rotate-fb15k-parallel-relation.yaml),[1@2](examples/experiments/fb15k/rotate-fb15k-distributed-relation.yaml)
stratification (CAR)  |   9.6s  |   **9.6min**  |   **0.6GB** |   **0.784**   |   [2@1](examples/experiments/fb15k/rotate-fb15k-parallel-stratification-car.yaml),[1@2](examples/experiments/fb15k/rotate-fb15k-distributed-stratification-car.yaml)
graph-cut  |   9.9s |   -   |   0.9GB   |   0.681   |   [2@1](examples/experiments/fb15k/rotate-fb15k-parallel-graph-cut.yaml),[1@2](examples/experiments/fb15k/rotate-fb15k-distributed-graph-cut.yaml)



#### Yago3-10

**ComplEx**

partition scheme    |   epoch time (1@2)  |   time to 95% MRR (1@2) |   data sent   |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
random  |   21.1s   |  66.9min |   14.5GB  |   **0.538**   |   [2@1](examples/experiments/yago3-10/complex-yago3-10-parallel-random.yaml),[1@2](examples/experiments/yago3-10/complex-yago3-10-distributed-random.yaml)
relation  |   23.6s |   41.3min |   13.6GB  |   **0.538**   |   [2@1](examples/experiments/yago3-10/complex-yago3-10-parallel-relation.yaml),[1@2](examples/experiments/yago3-10/complex-yago3-10-distributed-relation.yaml)
stratification (CAR)  |   **11.3s** |   **36.6min** |   0.8GB   |   0.531   |   [2@1](examples/experiments/yago3-10/complex-yago3-10-parallel-stratification-car.yaml),[1@2](examples/experiments/yago3-10/complex-yago3-10-distributed-stratification-car.yaml)
graph-cut  |   13.9s    |   -   |   **0.3GB**   |   0.211   |   [2@1](examples/experiments/yago3-10/complex-yago3-10-parallel-graph-cut.yaml),[1@2](examples/experiments/yago3-10/complex-yago3-10-distributed-graph-cut.yaml)

**RotatE**

partition scheme    |   epoch time (1@2)  |   time to 95% MRR (1@2) |   data sent   |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
random  |   44.1s   |   209.6min    |   6.9GB   |   0.437   |   [2@1](examples/experiments/yago3-10/rotate-yago3-10-parallel-random.yaml),[1@2](examples/experiments/yago3-10/rotate-yago3-10-distributed-random.yaml)
relation  |   53.0s |   265.2min    |   6.8GB   |   **0.441**   |   [2@1](examples/experiments/yago3-10/rotate-yago3-10-parallel-relation.yaml),[1@2](examples/experiments/yago3-10/rotate-yago3-10-distributed-relation.yaml)
stratification (CAR)  |   **43.3s** |   **176.8min**    |   0.6GB   |   0.438   |   [2@1](examples/experiments/yago3-10/rotate-yago3-10-parallel-stratification-car.yaml),[1@2](examples/experiments/yago3-10/rotate-yago3-10-distributed-stratification-car.yaml)
graph-cut  |   43.3s    |   -   |   **0.3GB**   |   0.336   |   [2@1](examples/experiments/yago3-10/rotate-yago3-10-parallel-graph-cut.yaml),[1@2](examples/experiments/yago3-10/rotate-yago3-10-distributed-graph-cut.yaml)


#### Wikidata5m

**ComplEx**

partition scheme    |   epoch time (1@2)  |   time to 95% MRR (1@2) |   data sent   |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
random  |   347.2s  |   173.6min    |   181.0GB |   0.296   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-random.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-random.yaml)
relation  |   320.5s    |   160.2min    |   178.1GB |   0.296   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-relation.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-relation.yaml)
stratification (CAR)  |   **228.0s**  |   **76.0min** |   14.4GB  |   **0.308**   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-stratification-car.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-stratification-car.yaml)
graph-cut  |   317.2s   |   -   |   **9.9GB**   |   0.192   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-graph-cut.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-graph-cut.yaml)

**RotatE**

partition scheme    |   epoch time (1@2)  |   time to 95% MRR (1@2) |   data sent   |   MRR |   config
--------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
random  |   501.2s  |   125.5min    |   82.3GB  |   0.256   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-random.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-random.yaml)
relation  |   484.5s    |   121.1min    |   79.6GB  |   0.259   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-relation.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-relation.yaml)
stratification (CAR)  |   **477.7** |   **79.6**    |   16.8GB  |   **0.264**   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-stratification-car.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-stratification-car.yaml)
graph-cut  |   496.4s   |   -   |   **10.5GB**  |   0.225   |   [2@1](examples/experiments/wikidata5m/complex-wikidata5m-parallel-graph-cut.yaml),[1@2](examples/experiments/wikidata5m/complex-wikidata5m-distributed-graph-cut.yaml)

#### Freebase


**ComplEx**

setup   |   partition scheme    |   epoch time  |   data sent   |   sMRR-1000   |   MRR |   config
-----   |   --------------- |   ------: |   ---------: |   -------: |   -----:  |   ------  |
2@2 |   random  |   1715.7s |   1564.0GB    |   0.829   |   0.356   |   [config](examples/experiments/freebase/complex-freebase-distributed-random-2@2.yaml)
2@2 |   relation  |   1658.7s   |   1496.2GB    |   0.824   |   0.332   |   [config](examples/experiments/freebase/complex-freebase-distributed-relation-2@2.yaml)
2@2 |   stratification (CAR)  |   **1105.9s**   |   277.8GB |   0.803   |   **0.477**   |   [config](examples/experiments/freebase/complex-freebase-distributed-stratification-car-2@2.yaml)
2@2 |   graph-cut  |   1691.0s  |   **113.2GB** |   0.801   |   0.268   |   [config](examples/experiments/freebase/complex-freebase-distributed-graph-cut-2@2.yaml)
4@2 |   stratification (CAR)  |   765.9s    |   312.2GB |   0.798   |   0.475   |   [config](examples/experiments/freebase/complex-freebase-distributed-stratification-car-4@2.yaml)
