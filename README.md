# Co-salient-feature-extraction-for-WRD
The official repo of the paper Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection.

## Environment Requirement
create enviroment and intall as following: pip install -r requirements.txt

## Data Format
trainset: NWRD_train
testset: NWRD_val, NWRD_test
Put the NWRD_train, NWRD_val and NWRD_test datasets to DCFM/data as the following structure:
Co-salient-feature-extraction-for-WRD
&nbsp;&nbsp;├── other codes  
&nbsp;&nbsp;├── ...  
&nbsp;&nbsp;└── data  
&nbsp;&nbsp;&nbsp;&nbsp;├── CoCo-SEG (CoCo-SEG's image files)  
&nbsp;&nbsp;&nbsp;&nbsp;├── CoCA (CoCA's image files)  
&nbsp;&nbsp;&nbsp;&nbsp;├── CoSOD3k (CoSOD3k's image files)  
&nbsp;&nbsp;&nbsp;&nbsp;└── Cosal2015 (Cosal2015's image files)  


## Trained model
trained model can be downloaded from [papermodel](https://drive.google.com/drive/folders/1kvPTjDiOU6_puIWmNVoKYcoLIxuSIz82?usp=sharing).
Run test.py for inference.

## Usage
Download pretrainde backbone model VGG.
Run train.py for training.

## Others
The code is based on DCFM. 
