# ETGraph
Data and source code of ETGraph dataset.

Due to the large size of the dataset, it has been uploaded to Google Drive for easy access and downloading.

- On-chain Ethereum transaction data in DGL graph format:
  
  https://drive.google.com/file/d/1lY2IC3LeRevSPZCE2U9o8taunGLEV0oY/view?usp=sharing 
- DGL graphs for wash-trading Ethereum addresses detection:

  Training graph: https://drive.google.com/file/d/14gW0EovXYiXd62XJayGGYRBEeZ3TTU4i/view?usp=sharing

  Validation graph: https://drive.google.com/file/d/1Qk1unpx0D66STS8GLjQNEsrOqngNQtSy/view?usp=sharing

  Test graph: https://drive.google.com/file/d/1JgoBdaD2e3wlvZmT-GPEpwEiFwtrPwSB/view?usp=sharing
- Off-chain Twitter data in DGL graph format:

  https://drive.google.com/file/d/1SNOg3QYoVWFRIl91o0tCeA4dReKwtEa0/view?usp=sharing



This repository contains three code folders and two key data files. The folders are organized as follows:

1. Ethereum link prediction
2. Wash trading address detection
3. Matching link prediction

The data files included in this repository are:

1. Twitter handles with their matching Ethereum addresses from OpenSea. (This is for the review. Formal version will not contain it as the information is already contained in the off-chain Twitter data in DGL graph format without leaking any personal information.)
2. Wash-trading transaction records collected from Dune.

This dataset is under license CC BY-NC-SA.
