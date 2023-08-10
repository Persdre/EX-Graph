# ETGraph
ETGraph is a rich dataset containing on-chain Ethereum transaction data and off-chain Twitter data in DGL graph format. It's aimed at various tasks such as Ethereum link prediction, wash-trading address detection, and matching link prediction.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Downloading the Dataset](#downloading-the-dataset)
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

## Getting Started
### Installation

1. Download this repository and extract the datasets to a desired location.
2. Create a Conda environment using the provided `environment.yml` file, which includes the requirements to run experiments in this repository. Run:

   ```
   conda env create -f environment.yml
   ```

### Requirements
We list main requirements of this repository below. For full requirements, please refer to the provided `environment.yml` file

- networkx>=2.8.3
- numpy>=1.22.3
- outdated>=0.2.1
- pandas>=1.4.2
- patool>=1.12
- requests>=2.27.1
- setuptools>=60.2.0
- torch>=1.11.0
- torch_scatter>=2.0.9

### Data Loading

e.g.
```
import dgl
graph = dgl.load_graphs("path/to/your/data.bin")
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
