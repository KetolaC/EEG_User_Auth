## Import libraries

from sklearn import ensemble as ensemblesk
from cuml import ensemble
from imblearn.over_sampling import SMOTE
import numpy as np
import csv
import os

debug = True  # Set debug = True to get debug messages in the console

gen_user = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']  # Set the genuine user

all_results = []

## Import the .csv files for training data, validation data, and testing data
#
#  This section imports the training, validation, and testing data from .csv
#  files and converts the data into list formats. If debug is true, the data
#  contained in the list will be printed out.
#
#
for p in gen_user:

    gen = p

    all_results.append(gen)

    cwd = os.getcwd()  # Get the current working directory (cwd)

    train_file = cwd + '/WAY_EEG_GAL_split_data/TrainingData.csv'  # Get training data .csv path
    valid_file = cwd + '/WAY_EEG_GAL_split_data/ValidationData.csv'  # Get validation data .csv path
    test_file = cwd + '/WAY_EEG_GAL_split_data/TestingData.csv'  # Get testing data .csv path

    # Open the .csv files in Python

    train = open(train_file)  # Open training data
    train_data = csv.reader(train)  # Store training data as a csv.reader object
    next(train_data)  # Skip the headers
    train_data = list(train_data)  # Convert training data to list

    valid = open(valid_file)  # Open validation data
    valid_data = csv.reader(valid)  # Store validation data as a .csv object
    next(valid_data)  # Skip the headers
    valid_data = list(valid_data)  # Convert validation data to list

    test = open(test_file)  # Open the testing data
    test_data = csv.reader(test)  # Store the testing data as a .csv object
    next(test_data)  # Skip the headers
    test_data = list(test_data)  # Convert testing data to list

    if debug:  # Print .csv data if debugging is activated
        print("Size of training data: " + str(len(train_data)))
        print("Size of validation data: " + str(len(valid_data)))
        print("Size of testing data: " + str(len(test_data)))

    if debug:
        print("Files imported successfully")

    del [cwd, train_file, valid_file, test_file, train, valid, test]  # Remove excess variables

    ## Balancing the training data
    #
    #  Balancing is performing by upsampling data using SMOTE (Synthetic Minority
    #  Over-sampling Technique). The genuine user is selected from the list of
    #  available subjects, and the number of genuine samples are increased to
    #  match the number of imposter samples. Binary labels are created to label
    #  the genuine samples as '1's and the imposter samples as '0's.
    #
    #

    if debug:
        print("Starting balancing")

    train_data_feats = [channels[2:] for channels in train_data]  # Remove original labels from dataset
    train_feats = []  # Create a list to store the converted features

    for i in train_data_feats:
        train_feats.append([float(j) for j in i])  # Convert features from char to float

    train_labels = []  # Create a list to store the binary labels

    for i in range(len(train_data)):
        train_labels.append(1) if train_data[i][0] == gen else train_labels.append(0)  # Create labels for training data

    smote_model = SMOTE(random_state=56)  # RNG seed randomly selected as 56 for replicability
    smote_feats, smote_labels = smote_model.fit_resample(train_feats, train_labels)  # Create upsampled data and labels

    if debug:
        print("Done balancing the data")
        print("Rows: " + str(len(smote_feats)) + ", Columns: " + str(len(smote_feats[0])))

    del [train_data, train_data_feats, train_feats, train_labels, smote_model]

    ## Label and convert the validation data
    #
    #  This section creates binary labels for the validation data, which assigns
    #  a '1' to genuine samples and a '0' to imposter samples. The feature data
    #  is converted from a char datatype to a float datatype for use in the SVM
    #  model.
    #

    if debug:
        print("Starting labeling of validation data")

    valid_data_feats = [channels[2:] for channels in valid_data]  # Remove original labels from dataset
    valid_feats = []  # Create a list to store the converted features
    valid_labels = []  # Create a list to store the binary labels

    for i in range(len(valid_data)):
        valid_labels.append(1) if valid_data[i][0] == gen else valid_labels.append(0)  # Create labels for training data

    if debug:
        print("Labeling completed. Feature conversion started.")

    for i in valid_data_feats:
        valid_feats.append([float(j) for j in i])  # Convert features from char to float

    if debug:
        print("Done converting validation features")

        del [valid_data, valid_data_feats]

    ## Label and convert the test data
    #
    #  This section creates binary labels for the testing data, which assigns a
    #  '1' to genuine samples and a '0' to imposter samples. The feature data is
    #  converted from a char datatype to a float datatype for use in the SVM
    #  model.
    #
    #

    if debug:
        print("Starting labeling of testing data")

    test_data_feats = [channels[2:] for channels in test_data]  # Remove original labels from dataset
    test_feats = []  # Create a list to store the converted features
    test_labels = []  # Create a list to store the binary labels

    if debug:
        print("Labeling completed. Feature conversion started.")

    for i in range(len(test_data)):
        test_labels.append(1) if test_data[i][0] == gen else test_labels.append(0)  # Create labels for training data

    for i in test_data_feats:
        test_feats.append([float(j) for j in i])  # Convert features from car to float

    if debug:
        print("Done converting test features")

        del [test_data, test_data_feats, gen]

    ## Train and validate the RF classifier with all features as a benchmark
    #
    #  In this section an SVM model is used with 4 different regularization
    #  parameters (0.1, 1, 10, 100), and two different gamma values (scale and
    #  auto). All combinations of these parameters are trained with the training
    #  data, and validated with the validation data. The data used in his section
    #  contains all of the originally extracted features, and will be used as a
    #  benchmark to which the feature reduction process will be compared.
    #
    #

    if debug:
        print("Starting benchmark RF testing")

    rf_model = ensemble.RandomForestClassifier(max_depth=-1)  # Create a RF model
    rf_model.fit(smote_feats, smote_labels)  # Fit the model using training data with all features
    print("Done fitting model")
    rf_pred = rf_model.predict(valid_feats)  # Predicts class of validation data
    score = rf_model.score(valid_feats, valid_labels)  # Gets the classification accuracy

    if debug:
        print("Completed benchmark RF testing")
        print("Results: " + str(score))

    print("\n\n")

    all_results.append(score)

    del [valid_labels, valid_feats, score]

    ## Channel Ranking
    #
    #  In this section, decision trees are used to rank the importance of
    #  each channels data
    #
    #

    if debug:
        print("Channel ranking started")

    dt_model = ensemblesk.ExtraTreesClassifier()  # Creates a dt classifier of 100 random decision trees
    dt_model.fit(smote_feats, smote_labels)  # Fits the dt model with the upsampled training data
    ranked_channels = dt_model.feature_importances_

    if debug:
        print("Channel ranking completed")
        print(ranked_channels)

    ## Test RF effeciency using the channel rankings

    if debug:
        print("Setting up variables for channel reduction")

    scores = []  # Creating a list to hold the RF scores for each channel reduction
    ranked_copy = ranked_channels.tolist()  # Create a copy of ranked channels
    feats_copy = np.array(smote_feats)  # Create a copy of the training features
    test_copy = np.array(test_feats)  # Create a copy of the test features

    all_results.append(ranked_channels)

    del [smote_feats, test_feats, ranked_channels]

    ##

    if debug:
        print("RF with channel reduction testing begins")

    for n in range(1, 32, 1):

        lowest = 1.00  # Keep track of the lowest ranked channel
        index = None

        for i in ranked_copy:
            if i < lowest:
                lowest = i
                index = ranked_copy.index(i)

        ranked_copy.remove(lowest)

        if debug:
            print(feats_copy.shape)

        feats_copy = np.delete(feats_copy, index, axis=1)
        test_copy = np.delete(test_copy, index, axis=1)

        if debug:
            print(feats_copy.shape)

        if debug:
            print("Starting RF testing with  " + str(32 - n) + " channels")

        rf_model = ensemble.RandomForestClassifier(n_jobs=-1)  # Create a RF model with worst channel removed
        rf_model.fit(feats_copy, smote_labels)  # Fit the model using training data with all features

        if debug:
            print("Done fitting model")

        rf_pred = rf_model.predict(test_copy)  # Predicts class of validation data
        scores.append(rf_model.score(test_copy, test_labels))  # Gets the classification accuracy

    if debug:
        print("Completed benchmark RF testing")
        print("Results: ")
        print(scores)

    all_results.append(scores)

    del [feats_copy, test_copy, test_labels, smote_labels, scores]
