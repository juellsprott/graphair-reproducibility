# Reproducibility Study Of Learning Fair Graph Representations Via Automated Data Augmentations
Repository containing code of "Reproducibility Study Of Learning Fair Graph Representations Via Automated Data Augmentations" for TMLR submission.

## Installing requirements

Create a virtual environment and install the required packages:

```
virtualenv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

For viewing and analyzing reports of runs, we utilize Weights and Biases. In order to run and view results post-run, having access to a Weights and Biases account is required. 

## Training and evaluating models 

Aside from Graphair, this repository has access to reimplementations for the FairAdj and FairDrop models. These models are used as baselines for our reproducibility study. The models are based on the [On Dyadic Fairness: Exploring and Mitigating Bias in Graph Connections](https://openreview.net/pdf?id=xgGS6PmzNq6) and [FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning](https://arxiv.org/abs/2104.14210) papers, respectively. The original repositories for the public FairDrop and FairAdj implementations can be found [here](https://github.com/ispamm/FairDrop) and [here](https://github.com/brandeis-machine-learning/FairAdj).

Running the models with hyperparameters obtained from grid searches can be done as follows:

```
python -m main --use_best_params --dataset <dataset> <model>
```

For node classification, the following options are available:

| Models   | Datasets         |
|----------|------------------|
| graphair | nba              |
|          | pokec_n          |
|          | pokec_z          |

For link prediction:

| Models   | Datasets         |
|----------|------------------|
| lpgraphair | citeseer             |
| fairdrop | cora       |
| graphair | pubmed  |

For example, in order to run link prediction task on the Cora dataset with Graphair with the best obtained hyperparameters, use the following command:


```
python -m main --use_best_params --dataset cora lpgraphair
```

For obtaining plots for claim 2 of our reproducibility study (node classification on Graphair only), use the  `--homophily` flag.

### Running grid search or custom hyperparameters

To test the model using your own hyperparameters, you can run the file in terminal using the following format:

```
python -m main --<generic_args> --dataset <dataset> <model> --<model_specific_args>
```

For training and evaluating the various models, we make use of Python's argparse library. In order to select and use a specified model, we use subparsers. To effectively use these subparsers, one must first specify any model-agnostic flags (denoted as generic_args), such as model learning rate, epochs and dataset, before specifying the model-specific arguments (denoted as model-specific-args)

Arguments denoted with '*' accept multiple values, in order to chain runs and perform a grid search (denoted as a ['sweep' in Weights and Biases'](https://docs.wandb.ai/guides/sweeps))

The generic arguments are as follows:

- `--epochs` (Number of epochs, default: `[500]`)*
- `--dataset` (Dataset name, default: `"pokec_z"`, choices: `["pokec_z", "pokec_n", "nba", "citeseer", "cora", "pubmed"]`)
- `--model_lr` (Model learning rate, default: `[1e-4]`)*
- `--use_best_params` (Ignore input args, use best hyperparameters, default: `False`)

For Graphair (both variants), the following arguments are available:

- `--alpha` (List of alpha values, default: `[1.0]`)*
- `--beta` (List of beta values, default: `[1.0]`)*
- `--gamma` (List of gamma values, default: `[1.0]`)*
- `--lam` (List of lambda values, default: `[1.0]`)*
- `--classifier_lr` (Classifier learning rate, default: `[1e-3]`)*
- `--wd` (Weight decay, default: `1e-5`)
- `--classifier_hidden` (Classifier hidden size, default: `[128]`)*
- `--model_hidden` (Model hidden size, default: `[64]`)*
- `--test_epochs` (Number of test epochs, default: `500`)
- `--disable_ep` (Specify whether to disable the ep component in graphair, accepts multiple boolean values, default: `[False]`)*
- `--disable_fm` (Specify whether to disable the fm component in graphair, accepts multiple boolean values, default: `[False]`)*
- `--homophily` (Specify whether claim 3 (plotting homophily and spearman) should be enabled or not, default: `False`)

For Fairdrop, the following arguments are available:

- `--hidden_gcn` (Hidden size for GCN, default: `[64]`)*
- `--hidden_gat` (Hidden size for GCN, default: `[64]`)*
- `--heads` (Number of heads, used for GAT, default: `[1]`)*
- `--model_type` (Model type for Fairdrop, either GCN or GAT, default: `['GCN']`)*
- `--delta` (Delta parameter for Fairdrop, default: `[0.16]`)*

For FairAdj, the following arguments are available:

- `--hidden1` (Hidden size for layer 1 of GCNModelVAE, default: `[32]`)*
- `--hidden2` (Hidden size for layer 2 of GCNModelVAE, default: `[16]`)*
- `--dropout` (Dropout rate, calculated as 1 - keep_prob, default: `[0.0]`)*
- `--outer_epochs` (Number of epochs to train, default: `[4]`)*
- `--eta` (Learning rate for adjacency matrix, default: `[0.2]`)*
- `--T1` (Default: `[50]`)*
- `--T2` (Default: `[20]`)*

## Acknowledgments

A large thanks to the original authors of the [Learning Fair Graph Representations via Automated Data Augmentations](https://openreview.net/forum?id=1_OGWcP1s9w) paper, whom provided a public implementation of their method in the [Dive into Graphs library](https://github.com/divelab/DIG), and supporting us with extensive communications during our research. 

Additionally, a big thanks to the authors of [On Dyadic Fairness: Exploring and Mitigating Bias in Graph Connections](https://openreview.net/pdf?id=xgGS6PmzNq6) and [FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning](https://arxiv.org/abs/2104.14210) for providing open source implementations of their models. 

