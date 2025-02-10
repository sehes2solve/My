# Machine Learning Based Propagation Loss (MLPL) Model for the Network Simulator 3 (ns-3)

The Machine Learning based Propagation Loss (MLPL) model is a module for the [network simulator 3 (ns-3)](https://www.nsnam.org) that uses machine learning to train a propagation loss model to reproduce the physical conditions of an experimental testbed. The propagation loss model is trained with network traces collected in the experimental testbed that shall be reproduced.



## ML Propagation Loss Models

The MLPL module contains two models depending on the method to train and predict the propagation loss.

* D-MLPL: Distance-based ML Propagation Loss Model
  * Train and predict the propagation loss according to the distance between the nodes.
* P-MLPL: Position-based ML Propagation Loss Model
  * Train and predict the propagation loss according to the positions of the nodes.

They are both implemented in the same set of files. The term MLPL is used to refer the generic model, regardless of the specific sub-model used. The specific sub-models are referred by their acronyms.

## Project Structure

The project is structured as follows:

* `datasets/`: Datasets to train the MLPL model
* `examples/`: MLPL's ns-3 examples
* `doc/`: Model's documentation
* `ml_model/`: Scripts to train and run the MLPL's ML models
* `model/`: MLPL's ns-3 model
* `test/`: MLPL's ns-3 test suite

## MLPL Setup

The instructions to set-up and run this project are explained below.

### Setup ns-3 and Dependencies

1. Download the latest version of [ns-3](https://gitlab.com/nsnam/ns-3-dev).

  ```shell
  git clone https://gitlab.com/nsnam/ns-3-dev.git
  ```

2. Download the [ns3-ai](https://apps.nsnam.org/app/ns3-ai/) app from the ns-3 App Store and install it in the `contrib/` subdirectory. In order to use ns-3.36+, please switch to the `cmake` branch.

  ```shell
  git clone --branch=cmake https://github.com/hust-diangroup/ns3-ai.git
  mv ns3-ai/ ns-3-dev/contrib/
  ```

3. add this folder to `contrib/` folder.

### Setup Python

As a best practice, it is recommended to use the a PyEnv (venv) virtual environment of this project. This separates and isolates Python environment where all dependencies are installed without interfering with other existing environments (including the base environment).


To activate the new virtual environment, run the following command:

```shell
source venv/bin/activate
```
The Python dependencies are registered in the [`requirements.txt`](requirements.txt) file. To install them, run the following command: (Note : they are aleready instaled )
```shell
pip install -r requirements.txt
```

### Build the ns-3 ML Propagation Loss Model

To build the ns-3 ML Propagation Loss model, run the following commands:

```shell
./ns3 configure
./ns3 build
```
### MLPL Datasets

This folder contains the datasets to train the ML models of the path loss and fast-fading.

The full documentation about the datasets is available in this page: [Datasets](../doc/datasets.md).


## MLPL Model Training and Usage

The MLPL model training and usage in ns-3 are explained below.

### Datasets

The datasets used to train the MLPL model are stored in the [datasets](datasets) directory

### ML Model Training

The scripts to train the ML models are located in the [`ml_model/`](ml_model/) directory. The documentation about training and running the ML models supporting the MLPL model are available in the following page:

[ML Model Training](doc/ml-model-training.md)

### MLPL Model Usage in ns-3

The documentation about using the MLPL model in ns-3 is available in the following page:

[MLPL Usage in ns-3](doc/mlpl-usage-ns3.md).

### MLPL Tests

This module contains tests that check that the module is working as expected. The tests are located in the directory `tests/`.

To run the tests, first open a terminal to activate the ML models, as explained in [MLPL Usage in ns-3](doc/mlpl-usage-ns3.md).

```shell
python ml_model/run_ml_propagation_loss_model.py  --dataset=position-dataset-example --mlpl-model=position --ml-algorithm=xgb
```

Then, run the tests in another terminal with the following command:

```shell
./test.py -s "ml-propagation-loss-model"
```

NOTE: Currently, the test suite only works with the parameters stated in the command above.

### MLPL Examples

The example `examples/ml-propagation-loss-throughput.cc` calculates the expected throughput between two nodes using the MLPL model for calculating the propagation loss.

To run the example, first open a terminal to activate the ML models, as explained [MLPL Usage in ns-3](doc/mlpl-usage-ns3.md).

```shell
python ml_model/run_ml_propagation_loss_model.py --dataset=DATASET --mlpl-model=MLPL_MODEL --ml-algorithm=ML_ALGORITHM
```

Then, run the example in another terminal with the following command:

```shell
./ns3 run "ml-propagation-loss-model-throughput --dataset=DATASET --lossModel=LOSS_MODEL"
```

The arguments of the scripts correspond to the ML model that shall be run and allow the selection of the dataset, the MLPL model and the ML algorithm.

## Published Papers

The list of papers published in international conferences and journals about the MLPL model is available in the following page:

[List of Published Papers](doc/published-papers.md)


* Eduardo Nuno Almeida, Helder Fontes, Rui Campos, and Manuel Ricardo. 2023. Position-Based Machine Learning Propagation Loss Model Enabling Fast Digital Twins of Wireless Networks in ns-3. In Proceedings of the 2023 Workshop on ns-3 (WNS3 '23). ACM, 69–77. <https://doi.org/10.1145/3592149.3592150>

  🏆 Best Paper Award
  🏆 ACM Artifacts Available Badge

  <details>
  <summary>BibTex</summary>

  ```bibtex
  @inproceedings{almeida2023position,
    author = {Almeida, Eduardo Nuno and Fontes, Helder and Campos, Rui and Ricardo, Manuel},
    title = {Position-Based Machine Learning Propagation Loss Model Enabling Fast Digital Twins of Wireless Networks in ns-3},
    year = {2023},
    isbn = {9798400707476},
    publisher = {ACM},
    booktitle = {Proceedings of the 2023 Workshop on ns-3},
    pages = {69–-77},
    doi = {10.1145/3592149.3592150},
  }
  ```

  </details>
