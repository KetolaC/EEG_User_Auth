# EEG_User_Auth Repository
This repository uses the WAY_EEG_GAL public EEG dataset [1] and performs user authentication using machine learning algorithms. 

Feature extraction and signal preprocessing is performed using Matlab. The Signal Processing Toolbox is required to run the Matlab code. A computer with 16 GB DDR is recommended for feature extraction.

## Instructions:

1. Clone the repository using the command **TODO**
```
$ git clone https://github.com/KetolaC/EEG_User_Auth.git
```

2. Enter the Feature_Selection folder and extract the data in the P1-P12 subfolders

3. Run preprocessing.m for each participant by changing **p = 1** on line 8 to the number of each participant form 1-12

```
p = 1;                                          % Choose the participant
```

4. Move the newly generated csv files obtained from preprocessing.m to the respective subfolders P1 - P12 in the Data_Splitting folder

## Citations:

[1] David Luciw, Matthew; Jarocka, Ewa; Edin, Benoni (2014): WAY-EEG-GAL: Multi-channel EEG Recordings During 3,936 Grasp and Lift Trials with Varying Weight and Friction. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.988376 
