# <img src="docs/source/images/logo/libkge-header-2880.png" alt="Distributed LibKGE: A knowledge graph embedding library" width="80%">

**NOTE: The preparation of this repository is still ongoing**


This is the code and configuration accompanying the paper "Parallel Training of Knowledge Graph Embedding Models: A Comparison of Techniques".
The code extends the knowledge graph embedding library [LibKGE](https://github.com/uma-pi1/kge).
For documentation on LibKGE refer to LibKGE repository.

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

dataset |   MRR |   config
------  |   ----:   |   ----
FB15k   |   0.804   |   [config](examples/experiments/fb15k/complex-fb15k-sequential.yaml)
Yago3-10    |   0.542   |   [config](examples/experiments/yago3-10/complex-yago3-10-sequential.yaml)
Wikidata    |   0.297   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-sequential.yaml)
Freebase (main memory)    | 0.344   |   [config](examples/experiments/freebase/complex-freebase-sequential.yaml)


**RotatE**

dataset |   MRR |   config
------  |   ----:   |   ----
FB15k   |   0.780   |   [config](examples/experiments/fb15k/rotate-fb15k-sequential.yaml)
Yago3-10    |   0.451   |   [config](examples/experiments/yago3-10/rotate-yago3-10-sequential.yaml)
Wikidata    |   0.258   |   [config](examples/experiments/wikidata5m/rotate-wikidata5m-sequential.yaml)
Freebase (main memory)    | 0.571   |   [config](examples/experiments/freebase/rotate-freebase-sequential.yaml)

### Multi-Machine Training
#### FB15k

**ComplEx**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.800   |   [config](examples/experiments/fb15k/complex-fb15k-distributed-random.yaml)
relation  |   0.800   |   [config](examples/experiments/fb15k/complex-fb15k-distributed-relation.yaml)
stratification (CAR)  |   0.799   |   [config](examples/experiments/fb15k/complex-fb15k-distributed-stratification-car.yaml)
graph-cut  |   0.601   |   [config](examples/experiments/fb15k/complex-fb15k-distributed-graph-cut.yaml)

**RotatE**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.774   |   [config](examples/experiments/fb15k/rotate-fb15k-distributed-random.yaml)
relation  |   0.774   |   [config](examples/experiments/fb15k/rotate-fb15k-distributed-relation.yaml)
stratification (CAR)  |   0.784   |   [config](examples/experiments/fb15k/rotate-fb15k-distributed-stratification-car.yaml)
graph-cut  |   0.681   |   [config](examples/experiments/fb15k/rotate-fb15k-distributed-graph-cut.yaml)



#### Yago3-10

**ComplEx**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.538   |   [config](examples/experiments/yago3-10/complex-yago3-10-distributed-random.yaml)
relation  |   0.538   |   [config](examples/experiments/yago3-10/complex-yago3-10-distributed-relation.yaml)
stratification (CAR)  |   0.531   |   [config](examples/experiments/yago3-10/complex-yago3-10-distributed-stratification-car.yaml)
graph-cut  |   0.211   |   [config](examples/experiments/yago3-10/complex-yago3-10-distributed-graph-cut.yaml)

**RotatE**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.437   |   [config](examples/experiments/yago3-10/rotate-yago3-10-distributed-random.yaml)
relation  |   0.441   |   [config](examples/experiments/yago3-10/rotate-yago3-10-distributed-relation.yaml)
stratification (CAR)  |   0.438   |   [config](examples/experiments/yago3-10/rotate-yago3-10-distributed-stratification-car.yaml)
graph-cut  |   0.336   |   [config](examples/experiments/yago3-10/rotate-yago3-10-distributed-graph-cut.yaml)


#### Wikidata5m

**ComplEx**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.296   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-random.yaml)
relation  |   0.296   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-relation.yaml)
stratification (CAR)  |   0.308   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-stratification-car.yaml)
graph-cut  |   0.192   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-graph-cut.yaml)

**RotatE**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.256   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-random.yaml)
relation  |   0.259   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-relation.yaml)
stratification (CAR)  |   0.264   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-stratification-car.yaml)
graph-cut  |   0.225   |   [config](examples/experiments/wikidata5m/complex-wikidata5m-distributed-graph-cut.yaml)

### Freebase multi-machine


**ComplEx**

partition scheme    |   MRR |   config
--------------- |   ------: |   ------
random  |   0.356   |   [config](examples/experiments/freebase/complex-freebase-distributed-random.yaml)
relation  |   0.332   |   [config](examples/experiments/freebase/complex-freebase-distributed-relation.yaml)
stratification (CAR)  |   0.477   |   [config](examples/experiments/freebase/complex-freebase-distributed-stratification-car.yaml)
graph-cut  |   0.268   |   [config](examples/experiments/freebase/complex-freebase-distributed-graph-cut.yaml)
