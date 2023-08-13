# ETGraph
ETGraph is a rich dataset containing on-chain Ethereum transaction data and off-chain Twitter data in DGL graph format. It's aimed at various tasks such as Ethereum link prediction, wash-trading address detection, and matching link prediction.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Downloading the Dataset](#downloading-the-dataset)
- [Dataset Schema](#dataset-schema)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Requirements](#requirements)
  - [Data Loading](#data-loading)
- [Detailed Structure and Usage](#detailed-structure-and-usage)
  - [Ethereum Link Prediction](#ethereum-link-prediction)
  - [Wash Trading Address Detection](#wash-trading-address-detection)
  - [Matching Link Prediction](#matching-link-prediction)
- [Tasks and Benchmarks](#tasks-and-benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Dataset Overview
ETGraph dataset consists of several components:

- **On-chain Ethereum Transaction Data**: Structured data in DGL graph format representing the Ethereum transactions.
- **Off-chain Twitter Data**: Complementary data from Twitter to provide additional insights.
- **Various Training, Validation, and Test Graphs**: Separated datasets for different tasks including wash trading address detection and link prediction.

## Downloading the Dataset
Due to the large size of the dataset, it has been uploaded to Google Drive for convenient downloading. Links for each part of the dataset are provided below:

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
| Twitter Graph                                  | <number of nodes>| <number of edges>| twitter_handle                      | <edge features>                            |
| Ethereum Graph                                 | <number of nodes>| <number of edges>| <node features>                     | from_id, to_id, weight, block_number       |
| Ethereum Graph with Twitter features           | <number of nodes>| <number of edges>| twitter_features, ethereum_features | combined_features                          |


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

### Data Loading

We suppose that you have downloaded the datasets to the corresponding task's folders. Then, you can use the following command to load the data in the pkl format.
```
import dgl
import pickle as pkl
graph = dgl.load_graphs("G_dgl.pkl")
```

## Detailed Structure and Usage

This section outlines the structure of the dataset and its application in various domains:

Ethereum Link Prediction: Explain how the dataset supports Ethereum link prediction tasks, including the features used, problem formulation, and potential methods to apply.
Wash Trading Address Detection: Detail the structure and features relevant for detecting wash trading addresses, and provide guidance on how to utilize this part of the dataset.
Matching Link Prediction: Describe how matching link prediction can be performed using the dataset, including the methodologies that can be employed and the expected outcomes.


## Tasks and Benchmarks

## Contributing

## License
This dataset is under license CC BY-NC-SA.
