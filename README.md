# Co-salient-feature-extraction-for-WRD
Semantic segmentation of wheat yellow/stripe rust disease images to segment out rust and non-rust pixels using supervised deep learning.

## Environment Requirement
create enviroment and intall as following: pip install -r requirements.txt

## Dataset
The NWRD dataset is a real-world segmentation dataset of wheat rust diseased and healthy leaf images specifically constructed for semantic segmentation of wheat rust disease. The NWRD dataset consists of 100 images in total at this moment.

Sample images from The NWRD dataset; annotated images showing rust disease along with their binary masks:
![image](https://github.com/user-attachments/assets/3e7eff0e-f546-4673-ac21-3c199af80628)
Dataset is available at: https://dll.seecs.nust.edu.pk/downloads/



## Data Format
trainset: NWRD_train
testset: NWRD_val, NWRD_test
Put the NWRD_train, NWRD_train and NWRD_test datasets to DCFM/data as the following structure:


Co-salient-feature-extraction-for-WRD

├── other codes  
├── ...  
└── data  
&nbsp;&nbsp;&nbsp;&nbsp;├── NWRD_train
&nbsp;&nbsp;&nbsp;&nbsp;├── NWRD_val 
&nbsp;&nbsp;&nbsp;&nbsp;└── NWRD_test


## Trained model
trained model can be downloaded from [papermodel](https://drive.google.com/drive/folders/1kvPTjDiOU6_puIWmNVoKYcoLIxuSIz82?usp=sharing).
Run test.py for inference.

## Others
The code is based on DCFM.
