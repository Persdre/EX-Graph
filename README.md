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
Download this repository and extract the contents to a desired location.

### Requirements
ETGraph depends on the following:

- networkx>=2.8.3
- numpy>=1.22.3
- outdated>=0.2.1
- pandas>=1.4.2
- patool>=1.12
- requests>=2.27.1
- setuptools>=60.2.0
- torch>=1.11.0
- torch_scatter>=2.0.9

``### Dataset Loading

```
import dgl
graph = dgl.load_graphs("path/to/your/data.bin")
```

## Installation
Provide instructions on how to install any dependencies, including code snippets for commonly used package managers, e.g.,


Due to the large size of the dataset, it has been uploaded to Google Drive for easy access and downloading.

- On-chain Ethereum transaction data in DGL graph format (also for Ethereum link prediction):
  
  https://drive.google.com/file/d/1lY2IC3LeRevSPZCE2U9o8taunGLEV0oY/view?usp=sharing 
- DGL graphs for wash-trading Ethereum addresses detection:

  Training graph without Twitter data: https://drive.google.com/file/d/14gW0EovXYiXd62XJayGGYRBEeZ3TTU4i/view?usp=sharing

  Training graph with Twitter data: https://drive.google.com/file/d/1Na3SmGzpo8zycQtTCtOQIWh_yJPOcwck/view?usp=sharing

  Validation graph without Twitter data: https://drive.google.com/file/d/1Qk1unpx0D66STS8GLjQNEsrOqngNQtSy/view?usp=sharing

  Validation graph with Twitter data: https://drive.google.com/file/d/1lBJt62hGL5E9W87LytqhOBohGxqJNds8/view?usp=sharing

  Test graph without Twitter data: https://drive.google.com/file/d/1JgoBdaD2e3wlvZmT-GPEpwEiFwtrPwSB/view?usp=sharing

  Test graph with Twitter data: https://drive.google.com/file/d/12F1P98oQNjh-gvUaoviPvfS_FLb4Xtpx/view?usp=sharing

- Off-chain Twitter data in DGL graph format (also for matching link prediction):

  https://drive.google.com/file/d/1SNOg3QYoVWFRIl91o0tCeA4dReKwtEa0/view?usp=sharing



This repository contains three code folders and two key data files. The folders are organized as follows:

1. Ethereum link prediction
2. Wash trading address detection
3. Matching link prediction

The data files included in this repository are:

1. Twitter handles (converted to numerical id for anonymization) with their matching Ethereum addresses from OpenSea. 
2. Wash-trading transaction records collected from Dune.

This dataset is under license CC BY-NC-SA.``
