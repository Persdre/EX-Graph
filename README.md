# ETGraph
ETGraph is a rich dataset containing on-chain Ethereum transaction data and off-chain Twitter data in DGL graph format. With a focus on bridging the gap between anonymous on-chain activities and identifiable off-chain information, it's specifically designed to facilitate various tasks. These include Ethereum link prediction, wash-trading address detection, and matching link prediction between Ethereum addresses and Twitter accounts. By integrating these two different dimensions, ETGraph provides a comprehensive resource for researchers and analysts to explore the complex relationships within the cryptocurrency landscape.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Downloading the Dataset](#downloading-the-dataset)
- [Dataset Schema](#dataset-schema)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Requirements](#requirements)
  - [Dataset Loading](#dataset-loading)
- [Using ETGraph](#Using-ETGraph)
  - [Ethereum Link Prediction](#ethereum-link-prediction)
  - [Wash Trading Address Detection](#wash-trading-address-detection)
  - [Matching Link Prediction](#matching-link-prediction)
- [License](#license)

## Dataset Overview
ETGraph dataset consists of several components:

- **On-chain Ethereum Transaction Data**: Structured data in DGL graph format representing the Ethereum transactions.
- **Off-chain Twitter Data**: Complementary data from Twitter to provide additional insights.
- **Various Training, Validation, and Test Graphs**: Separated datasets for different tasks including wash trading address detection and link prediction.
- **Twitter-matching.csv**: Twitter accounts and the Ethereum addresses matched with them.
- **dune_wash_trade_tx.csv**: Wash-trading records collected from Dune.

## Downloading the Dataset

`twitter_matching.csv` and `dune_wash_trade_tx.csv` are in the repository. Due to the large size of other datasets, they have been uploaded to Google Drive for convenient downloading. Links for each part of the dataset are provided below:

- Twitter graph
- Ethereum graph
- Ethereum graph with Twitter features
- Wash-trading Addresses detection graph

- [On-chain Ethereum transaction data](https://drive.google.com/file/d/1lY2IC3LeRevSPZCE2U9o8taunGLEV0oY/view?usp=sharing)
- [DGL graphs for wash-trading Ethereum addresses detection](https://drive.google.com/file/d/14gW0EovXYiXd62XJayGGYRBEeZ3TTU4i/view?usp=sharing)
- [Validation and Test Graphs with and without Twitter Data](#)
  - [Training graph with Twitter data](https://drive.google.com/file/d/1Na3SmGzpo8zycQtTCtOQIWh_yJPOcwck/view?usp=sharing)
  - [Validation graph without Twitter data](https://drive.google.com/file/d/1Qk1unpx0D66STS8GLjQNEsrOqngNQtSy/view?usp=sharing)
  - [Test graph without Twitter data](https://drive.google.com/file/d/1JgoBdaD2e3wlvZmT-GPEpwEiFwtrPwSB/view?usp=sharing)
- [Off-chain Twitter data in DGL graph format](https://drive.google.com/file/d/1SNOg3QYoVWFRIl91o0tCeA4dReKwtEa0/view?usp=sharing)

## Dataset Schema
| Dataset                                         | Nodes            | Edges            | Node Features                        | Edge Features                              |
|------------------------------------------------|------------------|------------------|-------------------------------------|--------------------------------------------|
| Twitter Graph                                  | <number of nodes>| <number of edges>| twitter_handle, twitter_semantic_features                      | n.a.                            |
| Ethereum Graph                                 | <number of nodes>| <number of edges>| <node features>                     | from_id, to_id, weight, block_number       |
| Ethereum Graph with Twitter features           | <number of nodes>| <number of edges>| twitter_features, ethereum_features, combined_features | from_id, to_id, weight, block_number                          |
| Wash-trading Addresses Detection Graph           | <number of nodes>| <number of edges>| ethereum_features, twitter_semantic_features, twitter_structure_features, ethereum_twitter_combined_features | from_id, to_id, weight, block_number                          |


## Getting Started
### Installation

1. Download this repository and extract the datasets to a desired location.
2. Create a Conda environment using the provided `environment.yml` file, which includes the requirements to run experiments in this repository. Run:

   ```
   conda env create -f environment.yml
   ```

### Requirements
We list main requirements of this repository below. For full requirements, please refer to the provided `environment.yml` file

- dgl==1.1.0
- torch==2.0.0+cu117
- torch-cluster==1.6.1+pt20cu117
- torch-geometric==2.3.0
- torch-scatter==2.1.1+pt20cu117
- torch-sparse==0.6.17+pt20cu117
- torch-spline-conv==1.2.2+pt20cu117
- torchaudio==2.0.1+cu117
- torchmetrics==0.11.4
- torchvision==0.15.1+cu117
- transformers==4.30.2

### Dataset Loading

We suppose that you have downloaded the datasets to the corresponding task's folders. Then, you can use the following command to load the data in the pkl format.

```
import dgl
import pickle as pkl
with open('G_dgl_with_twitter_features_converted.pkl', 'rb') as f:
    G_dgl_with_twitter_features_converted = pkl.load(f)
```


## Using ETGraph

This section details how to using ETGraph to run benchmark baselines. We explain experiments Ethereum link prediction, wash-trading addresses detection, matching link prediction one by one.

### Ethereum Link Prediction
1. Navigate to ethereum_link_prediction folder, and download the Ethereum graph with Twitter features to this folder. 
2. Each experiment code file is named as `model_wo/with_twitter.py`. `wo` means this experiment does not consider Twitter features. `with` means this experiment considers Twitter features. For example, if you want to run clusterGCN model considering Twitter features in Ethereum link prediction, the command is as below:

    ``` 
    python clusterGCN_with_twitter.py
    ```

### Wash-trading Address Detection
1. Navigate to wash_trading_address_detection folder, and download the wash trading graph to this folder. 
2. Each experiment code file is named as `model_wo/with_twitter.py`. `wo` means this experiment does not consider Twitter features. `with` means this experiment considers Twitter features. For example, if you want to run GCN model considering Twitter features, the command is as below:
  
    ``` 
    python GCN_with_twitter.py
    ```

### Matching Link Prediction
1. Navigate to matching_link_prediction folder, and download the matching link graph to this folder. 
2. Each experiment code file is named as `model.py`. For example, if you want to run GCN model on this matching link prediction, the command is as below:
   
    ``` 
    python GCN.py
    ```

## License
This dataset is under license CC BY-NC-SA.
