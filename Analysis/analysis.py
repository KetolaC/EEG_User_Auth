#######################################################################################################################
##
##  The analysis library contains methods to extract results from the classifier and provide graphical outputs of the
##  results.
##
#######################################################################################################################

import re
import numpy
import numpy as np


class _ParticipantDictionary(dict):

    def __init__(self, *args, **kwargs):
        """
        A class to store participant data
        :param args: key
        :param kwargs: value
        """
        self.update(*args, **kwargs)    # Add inputs using the dict update function

    def __str__(self):
        s = ""
        for k, v in self.items():
            s = s + '\n' + k + ' : '    # Add newline and key
            buffer = ' ' * (len(k) + 3)

            for i in v:
                if i == v[0]:
                    s = s + str(i) + "\n"
                else:
                    s = s + buffer + str(i) + "\n"
        return s

    def __repr__(self):
        return "_ParticipantDictionary()"


class analysis:

    def __init__(self, result_path, channel_key=None):

        if not isinstance(result_path, str):
            raise ValueError("Result path must be a string containing a valid file path.")
        if not isinstance(channel_key, list) and channel_key!=None:
            raise ValueError("Channel key must be a list of strings, integers, or characters.")

        self._path = result_path                        # The filepath to the data
        self._key = channel_key                         # A key to label channels
        self._num_chn = None                            # The number of electrodes used
        self._par_data = []                             # The scores for every participant
        self._par = None                                # The label for each participant
        self._num_par = None                            # The total number of participants
        self._delta = []                                # The change between scores for channel reduction
        self.__readfile()                               # Read the contents of the provided file
        self.__setChannelParameters()                   # Determine the number of channels and channel key

    def __readfile(self):
        """
        A method to extract accuracy and gini importance scores incurred during channel reduction. \n
        :return: No object is returned
        """

        result_file = open(self._path, 'r')
        result_data = result_file.read()
        self._par = re.findall('Participant [0-9]*$', result_data, re.MULTILINE)  # Extract each participant
        self._num_par = len(self._par)

        for i in range(self._num_par):
            self._par_data.append([])

        ch_rankings = re.findall(r'Channel rankings: (.*$)', result_data, re.MULTILINE)     # Extracts rankings
        bm_rankings = re.findall(r'Benchmark Score: (.*$)', result_data, re.MULTILINE)   # Benchmark accuracies
        pr_rankings = re.findall(r'Ranked performance: (.*$)', result_data, re.MULTILINE)   # Extracts accuracies
        b_cl_time = re.findall(r'Classification Time: (.*$)', result_data, re.MULTILINE)    # Benchmarking times
        d_cl_times = re.findall(r'Classification Times: (.*$)', result_data, re.MULTILINE)    # Extracts times

        acc_scores = [', '.join([bm_rankings[ind], pr_rankings[ind]]) for ind in range(len(pr_rankings))]
        cl_times = [', '.join([b_cl_time[ind], d_cl_times[ind]]) for ind in range(len(d_cl_times))]


        for i in range(self._num_par):
            self._par_data[i].append([float(r) for r in acc_scores[i].split(',')])
            self._par_data[i].append([float(r) for r in ch_rankings[i].split(',')])
            self._par_data[i].append([float(r) for r in cl_times[i].split(',')])

        del [result_file, result_data, ch_rankings, pr_rankings, cl_times]

    def __setChannelParameters(self):
        """
        A method to determine the number of electrodes in use and assign undefined channel keys
        :return:
        """

        self._num_chn = len(self._par_data[0][0])

        if self._key == None:
            self._key = []
            for n in range(1, self._num_chn + 1):
                self._key.append('Channel ' + str(n))

    def __getScore(self, score=0):
        par_score = list()                              # Create a list to store score
        for par in self._par_data:                      # Parse through each participant
            par_score.append(par[score])                 # Attach score
        return par_score                                # Return a list of scores for all participants

    def __str__(self):
        pass

    def accuracy(self):
        """
        Retrieves the accuracy for every participant for every channel reduction.\n
        :return: A list of lists containing accuracy by participant
        """

        return self.__getScore()

    def giniImportance(self):
        """
      Retrieves the gini importance for every participant for every channel reduction.\n
      :return: A list of lists containing gini importance by participant
      """

        return self.__getScore(score=1)

    def classTimes(self):
        """
      Retrieves the classification time for every participant for every channel reduction.\n
      :return: A list of lists containing gini importance by participant
      """

        return self.__getScore(score=2)

    def delta(self, score=0):

        for p in range(self._num_par):
            diff = []
            for n in range(len(self._par_data[p][score]) - 1):
                diff.append(self._par_data[p][score][n+1]-self._par_data[p][score][n])
            self._delta.append(diff)

        return self._delta

    def average(self, gini=0):
        """
        The average function will return the average accuracy or gini importance of each channel. \n
        :param gini: When gini=1 the average gini importance is calculated. Otherwise, when gini=0 the average accuracy is calculated
        :return: The average score for each channel
        """

        avg = []                                        # Create a list to store the average accuracy each reduction
        for c in range(self._num_chn):                  # Go through all 32 channels
            avg_acc = 0                                 # Variable to store the average accuracy
            for i in range(self._num_par):              # Parse through the channel for all participants
                avg_acc = self._par_data[i][gini][c] + avg_acc  # Tally up the accuracies

            avg_acc = avg_acc / self._num_par           # Get the average accuracy
            avg.append(avg_acc)                         # Append the average accuracy for the channel to the list

        return avg

    def sortByParticipant(self, *args, score=0):
        par_dic = _ParticipantDictionary.fromkeys(self._par, [])    # Create a participant dictionary
        scores = [self.__getScore(score=score)]

        for band in args:
            scores.append(band.__getScore(score=score))

        n = 0

        for p in self._par:
            par_dic[p] = [ps[n] for ps in scores]
            n = n + 1

        return par_dic

    def rankChannels(self):
        top_chn = []
        temp_avg = self.average(gini=1)

        for c in range(self._num_chn):
            max_gini = temp_avg.index(max(temp_avg))  # Index of highest gini
            top_chn.append(self._key[max_gini])                              # Store top channel
            temp_avg[max_gini] = 0.00                                         # Set top channel to 0
            #
        return top_chn

if __name__ == "__main__":

    import os
    import matplotlib.pyplot as plt

    cwd = os.getcwd()  # Get the current working directory (cwd)

    # test_data = "example_timed.txt"   # Example results file for a timed system

    all_data = "../Classification/Results/all_results_timed.txt"  # Name of the file for all bands
    theta_data = "../Classification/Results/theta_results_timed.txt"  # Name of the file for theta band
    delta_data = "../Classification/Results/delta_results_timed.txt"  # Name of the file for delta band
    alpha_data = "../Classification/Results/alpha_results_timed.txt"  # Name of the file for alpha band
    beta_data = "../Classification/Results/beta_results_timed.txt"  # Name of the file for beta band
    gamma_data = "../Classification/Results/gamma_results_timed.txt"  # Name of the file for theta band

    placement = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
                 'PO10']  # The position of each channel in order

    all_bands = analysis(all_data, placement)
    theta_band = analysis(theta_data, placement)
    delta_band = analysis(delta_data, placement)
    alpha_band = analysis(alpha_data, placement)
    beta_band = analysis(beta_data, placement)
    gamma_band = analysis(gamma_data, placement)

    a_time = all_bands.classTimes()

    all_timing = all_bands.average(gini=0)
    theta_timing = theta_band.average(gini=0)
    delta_timing = delta_band.average(gini=0)
    alpha_timing = alpha_band.average(gini=0)
    beta_timing = beta_band.average(gini=0)
    gamma_timing = gamma_band.average(gini=0)

    avg_times = [all_timing, theta_timing, delta_timing, alpha_timing, beta_timing, gamma_timing]

    x_axis = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
              4, 3, 2, 1]  # Labels for the x-axis

    for i in avg_times:
        std = np.std(i, ddof=1)
        print(i)
        print("32 channel accuracy:", i[0], ", 14 channel accuracy:", i[18], ", Statistically significant threshold:",
              i[0]-0.015, ", Threshold met at: ", 32-i.index(list(filter(lambda j: j < i[0]-0.015, i))[0]),
              ", 14 channels more than 1% difference: ", i[18]<i[0]-0.015)

    # labels = ["All Bands", "Theta", "Delta", "Alpha", "Beta", "Gamma"]
    #
    # for band in avg_times:
    #     plt.plot(x_axis, band, '-o', label=labels[avg_times.index(band)])  # Plot data
    #     print(band)
    #
    # plt.ylabel("Average Time (s)")  # Label the y-axis
    # plt.xlabel("Channels Used")  # Label the x-axis
    # plt.title("Average Classifiction Time in Overall Band")  # Add a title
    # plt.legend(loc=1)  # Add a legend to the plot
    # fname = "test_avg_time.png"  # Name for the plot savefile
    # plt.savefig(fname)  # Save the plot
    # plt.show()
    #
    # placement = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    #              'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
    #              'PO10']  # The position of each channel in order
