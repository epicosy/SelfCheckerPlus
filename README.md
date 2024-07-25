# SelfCheckerPlus
This repository provides an improved version of the tool published in:

Y. Xiao et al., "Self-Checking Deep Neural Networks in Deployment," 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), Madrid, ES, 2021, pp. 372-384, doi: 10.1109/ICSE43902.2021.00044.


## Installation

1. Clone the repository:
```shell
$ git clone <repository-url>
$ cd <repository-directory>
```

2. Set up a virtual environment (optional but recommended):
```shell
$ python3.10 -m venv env
$ source env/bin/activate
```

3. Install the required dependencies:
```shell
$ pip install -r requirements.txt
```

### Usage

```
usage: main.py [-h] -m MODEL [-wd WORKDIR] [-bs BATCH_SIZE] [-oal] [-odl] {analyze,infer} ...
```

A self-checking tool for Deep Neural Networks to detect the potentially incorrect model decision and generate advice
to auto-correct the model decision on runtime.

positional arguments:
  {analyze,infer}

options:
-  -m MODEL, --model MODEL (Path to the model)
-  -wd WORKDIR, --workdir WORKDIR (Working directory)
-  -bs BATCH_SIZE, --batch_size BATCH_SIZE (Sets batch size - default is 128)
-  -oal, --only_activation_layers (Analyze only the dense layers)
-  -odl, --only_dense_layers (Analyze only the activation layers)


#### Analyze Command
Obtains the density functions for the combination of classes and layers and inferred classes.
It also performs the Layer selection for alarm and advice.


```shell
$ python -m selfchecker.main -m path/to/your/model.h5 -wd /path/to/working/directory analyze -tx path/to/train/features.npy -ty path/to/train/labels.npy -vx path/to/val/features.npy -vy path/to/val/labels.npy
```

##### Optional Arguments:
- --var_threshold, -var_threshold: Set the variance threshold (default is 1e-5).


#### Infer Command
Performs the alarm and advice analysis.

```shell
$ python selfcheckerplus.py -m path/to/your/model.h5 -wd /path/to/working/directory infer -tx path/to/test/features.npy -ty path/to/test/labels.npy
```
