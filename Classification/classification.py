## Import libraries

from sklearn import ensemble as ensemblesk
from cuml import ensemble
from imblearn.over_sampling import SMOTE
import numpy as np
import time
import os


# Create a classifier class

class Classifier:

    def __init__(self, debug=False, timer=True, result_filepath=None):
        self.debug = debug
        self.timer = timer
        self.filepath = result_filepath
        self.model = None
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.all_results = []
        self.gen_users = None

    def _set_model(self, model):
        models = {'rf': ensemble.RandomForestClassifier(),
                  'svm': None,
                  'knn': None,
                  'lda': None,
                  'xgb': None
                  }

        self.model = models.get(model, None)

    def _extract_labels(self):
        pass

    def _set_output_file(self):
        pass

    def load_csv(self, csv_path):
        pass

    def choose_model(self, model):
        if model in ['rf', 'svm', 'knn', 'lda', 'xgb']:
            self._set_model(model)
            print(self.model)
        else:
            raise ValueError("Unsupported model argument. Supported arguments are 'rf', 'svm', 'knn', 'lda', or 'xgb'")


    def binarize(self):
        pass

    def SMOTE(self):
        pass

    def validate(self):
        pass

    def rank_channels(self):
        pass

    def channel_reduction(self):
        pass


if __name__ == '__main__':

    cwd = os.getcwd()  # Get the current working directory (cwd)

    file = cwd + "/Results/theta_results.txt"

    test = Classifier(debug=True, result_filepath=file)

    # ## Setup initial variables for authentication
    # start_time = time.time()
    #
    # debug = True  # Set debug = True to get debug messages in the console
    #
    # gen_user= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Set the genuine user
    #
    # all_results = []
    #
    # ## Setup a text file to write results to
    #
    # cwd = os.getcwd()  # Get the current working directory (cwd)
    #
    # file = cwd + "/Results/theta_results.txt"
    #
    # ## Import the .csv files for training data, validation data, and testing data
    # #
    # #  This section imports the training, validation, and testing data from .csv
    # #  files and converts the data into list formats. If debug is true, the data
    # #  contained in the list will be printed out. To change the band, add Aplha_,
    # #  Beta_, Delta_, Gamma_, or Theta_ before TrainingData.csv, ValidationData.csv,
    # #  and TestingData.csv.
    # #  -------------------------------------------------------------------------------------------------------------  #
    #
    # train_file = cwd + '/WAY_EEG_GAL_split_data/Theta_TrainingData.csv'  # Get training data .csv path
    # valid_file = cwd + '/WAY_EEG_GAL_split_data/Theta_ValidationData.csv'  # Get validation data .csv path
    # test_file = cwd + '/WAY_EEG_GAL_split_data/Theta_TestingData.csv'  # Get testing data .csv path
    #
    # channels = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    #             29, 30, 31, 32, 33)
    #
    # train = np.loadtxt(train_file, dtype="float32", skiprows=1, delimiter=',', usecols=channels)
    # valid = np.loadtxt(valid_file, dtype="float32", skiprows=1, delimiter=',', usecols=channels)
    # test = np.loadtxt(test_file, dtype="float32", skiprows=1, delimiter=',', usecols=channels)
    #
    # if debug:  # Print .csv data size if debugging is activated
    #     print("Size of training data: " + str(train.shape))
    #     print("Size of validation data: " + str(valid.shape))
    #     print("Size of testing data: " + str(test.shape))
    #     print("Files imported successfully\n\n")
    #
    # ## seperate data into labels and feature data
    #
    # train_labels = train[:, 0]  # Save labels of training data
    # train_data = np.delete(train, 0, axis=1)  # Remove labels from feature dataset
    #
    # valid_labels = valid[:, 0]  # Save labels of validation data
    # valid_data = np.delete(valid, 0, axis=1)  # Remove labels from feature dataset
    #
    # test_labels = valid[:, 0]  # Save labels of validation data
    # test_data = np.delete(valid, 0, axis=1)  # Remove labels from feature dataset
    #
    # del [cwd, train_file, valid_file, test_file, channels, train, valid, test]  # Remove excess variables
    #
    # ## Parse through all participants
    #
    # for p in gen_user:
    #
    #     if debug:
    #         par_time = time.time()
    #
    #     gen = p
    #
    #     if debug:
    #         print("Participant " + str(p) + "\n")
    #
    #     all_results.append(gen)
    #
    #     to_write = "Participant " + str(gen) + "\n\n"
    #
    #     f = open(file, 'a')
    #     f.write(to_write)
    #     f.close()
    #
    #     # Binerizing Data
    #     #
    #     # Binary labels are created to label the genuine samples as '1's and the imposter samples as '0's. This allows for
    #     # binary classification to be performed.
    #     #  -------------------------------------------------------------------------------------------------------------  #
    #
    #     bin_train_labels = np.empty_like(train_labels, dtype=int)
    #     bin_valid_labels = np.empty_like(valid_labels, dtype=int)
    #     bin_test_labels = np.empty_like(test_labels, dtype=int)
    #
    #     for i in range(len(train_data)):
    #         bin_train_labels[i] = 1 if train_labels[i] == gen else 0  # Create labels for training data
    #
    #     for i in range(len(valid_data)):
    #         bin_valid_labels[i] = 1 if valid_labels[i] == gen else 0  # Create labels for validation data
    #         bin_test_labels[i] = 1 if test_labels[i] == gen else 0  # Create labels for validation data
    #
    #     del [gen, to_write, f]
    #
    #     # Balancing the training data
    #     #
    #     #  Balancing is performing by upsampling data using SMOTE (Synthetic Minority
    #     #  Over-sampling Technique). The genuine user is selected from the list of
    #     #  available subjects, and the number of genuine samples are increased to
    #     #  match the number of imposter samples.
    #     #  -------------------------------------------------------------------------------------------------------------  #
    #
    #     if debug:
    #         print("Starting balancing")
    #         smote_time = time.time()
    #
    #     smote_model = SMOTE(random_state=56, n_jobs=-1)  # RNG seed randomly selected as 56 for replicability
    #     smote_data, smote_labels = smote_model.fit_resample(train_data,
    #                                                         bin_train_labels)  # Create upsampled data and labels
    #
    #     if debug:
    #         print("Done balancing the data")
    #         print("Balancing time: %s seconds" % (time.time() - smote_time))
    #
    #     del [smote_time, smote_model, bin_train_labels]
    #
    #     ## Train and validate the RF classifier with all features as a benchmark
    #     #
    #     #  In this section an RF model is designed with different parameters. All combinations of these parameters are
    #     #  trained with the training data, and validated with the validation data. The data used in his section
    #     #  contains all the originally extracted features, and will be used as a benchmark to which the feature
    #     #  reduction process will be compared.
    #     #  -------------------------------------------------------------------------------------------------------------  #
    #
    #     # Setup the possible parameters for the RF model
    #
    #     n_trees = [50, 75]  # Number of trees
    #     split = 0  # 0 = gini impurity
    #     samples = 1.0  # Use all data for every tree
    #     depth = 64
    #
    #     if debug:
    #         print("Starting benchmark RF testing")
    #         bench_time = time.time()
    #
    #     best = 0.0
    #
    #     rf_model = ensemble.RandomForestClassifier(n_estimators=75, split_criterion=split, max_samples=samples,
    #                                                max_depth=depth, max_features='auto', random_state=56,
    #                                                n_streams=1)  # Create a RF model
    #     rf_model.fit(smote_data, smote_labels)  # Fit the model using training data with all features
    #
    #     rf_pred = rf_model.predict(valid_data)  # Predicts class of validation data
    #     score = rf_model.score(valid_data, bin_valid_labels)  # Gets the classification accuracy
    #
    #     if debug:
    #         print("Completed benchmark RF testing")
    #         print("Results: " + str(score))
    #         print("Benchmarking time: %s seconds" % (time.time() - bench_time))
    #
    #         print("\n\n")
    #
    #     all_results.append(score)
    #
    #     to_write = "Benchmark Score: " + str(score) + "\n\n"
    #
    #     f = open(file, 'a')
    #     f.write(to_write)
    #     f.close()
    #
    #     del [bench_time, bin_valid_labels, score, rf_model, to_write, f]
    #
    #     ## Channel Ranking
    #     #
    #     #  In this section, decision trees are used to rank the importance of each channels data. This will be used
    #     #  to determine which channels produce the features that are the most distinct.
    #     #  -------------------------------------------------------------------------------------------------------------  #
    #
    #     if debug:
    #         print("Channel ranking started")
    #         chann_time = time.time()
    #
    #     dt_model = ensemblesk.ExtraTreesClassifier(n_jobs=-1)   # Creates a dt classifier of 100 random decision trees
    #     dt_model.fit(smote_data, smote_labels)                  # Fits the dt model with the upsampled training data
    #     ranked_channels = dt_model.feature_importances_
    #
    #     # print("Setting up model")
    #     # chan_ranker = ensemblesk.RandomForestClassifier(random_state=56, n_jobs=-1)
    #     # print("Fitting model")
    #     # chan_ranker.fit(smote_data, smote_labels)
    #     # print("Getting ranked channels")
    #     # ranked_channels = chan_ranker.feature_importances_
    #
    #     if debug:
    #         print("Channel ranking completed")
    #         print(ranked_channels)
    #         print("Channel Ranking time: %s seconds" % (time.time() - chann_time))
    #
    #     all_results.append(ranked_channels)
    #
    #     chan_write = ', '.join(str(channel) for channel in ranked_channels)
    #     to_write = "Channel rankings: " + chan_write + "\n\n"
    #
    #     f = open(file, 'a')
    #     f.write(to_write)
    #     f.close()
    #
    #     del [chann_time, dt_model, chan_write, to_write, f]
    #
    #     ## Perform RF With Channel Reduction
    #     #
    #     #  Using the parameters determined in the validation step, models will be developed for authenticating individuals
    #     #  using less channels. The least important channel will be removed each round. The least important channel will
    #     #  be that with the lowest score as obtained from the channel ranking.
    #     #  -------------------------------------------------------------------------------------------------------------  #
    #
    #     # Setup Variables for channel reduction
    #
    #     if debug:
    #         print("RF with channel reduction testing begins")
    #         chred_time = time.time()
    #
    #     scores = np.array([])  # Creating an array to hold the RF scores for each channel reduction
    #     test_copy = np.array(test_data)  # Create a copy of the test features
    #
    #     for n in range(1, 32, 1):
    #
    #         if debug:
    #             curRF_time = time.time()
    #
    #         lowest = 1.00  # Keep track of the lowest ranked channel
    #         index = None
    #
    #         for i in ranked_channels:
    #             if i < lowest:
    #                 lowest = i
    #                 index = np.where(ranked_channels == i)
    #
    #         ranked_channels = np.delete(ranked_channels, index)
    #
    #         smote_data = np.delete(smote_data, index, axis=1)
    #         test_copy = np.delete(test_copy, index, axis=1)
    #
    #         if debug:
    #             print("Starting RF testing with  " + str(32 - n) + " channels")
    #
    #         rf_model = ensemble.RandomForestClassifier(n_estimators=75, split_criterion=split, max_samples=samples,
    #                                                    max_depth=depth, max_features='auto', random_state=56,
    #                                                    n_streams=1) # Create a RF model with worst channel removed
    #         rf_model.fit(smote_data, smote_labels)  # Fit the model using training data with all features
    #
    #         rf_pred = rf_model.predict(test_copy)  # Predicts class of validation data
    #         score = rf_model.score(test_copy, bin_test_labels)
    #
    #         scores = np.append(scores, score)  # Gets the classification accuracy
    #
    #         if debug:
    #             print("Done predicting data class")
    #             print("Execution time: %s seconds" % (time.time() - curRF_time))
    #
    #     if debug:
    #         print("Completed channel reduction testing")
    #         print("Results: ")
    #         print(scores)
    #         print("Execution time: %s seconds\n\n" % (time.time() - chred_time))
    #
    #     all_results.append(scores)  # Save accuracy scores
    #
    #     acc_write = ', '.join(str(acc) for acc in scores)
    #     to_write = "Ranked performance: " + acc_write + "\n\n"
    #
    #     f = open(file, 'a')
    #     f.write(to_write)  # Write accuracy scores to file
    #     f.close()
    #
    #     print("Total time for participant: %s seconds \n" % (time.time() - par_time))
    #
    #     del [smote_data, smote_labels, scores, test_copy, bin_test_labels, acc_write, to_write, f, rf_model,
    #          lowest, par_time]  # Remove excess variables
    #
    # if debug:
    #     print("Execution time: %s seconds" % (time.time() - start_time))
    #
    # print(all_results)
