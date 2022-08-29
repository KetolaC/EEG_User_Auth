#######################################################################################################################
##
##  The analysis library contains methods to extract results from the classifier and provide graphical outputs of the
##  results.
##
#######################################################################################################################

import re

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
        self._num_chn = 0                               # The number of electrodes used
        self._par_data = []                             # The scores for every participant
        self._par = None                                # The label for each participant
        self._num_par = 0                               # The total number of participants
        self.__readfile()                               # Read the contents of the provided file
        self.__setChannelParameters()                   # Determine the number of channels and channel key

    def __readfile(self):
        """
        A method to extract accuracy and gini importance scores incurred during channel reduction. \n
        :return: No object is returned
        """

        result_file = open(self._path, 'r')
        result_data = result_file.read()
        self._par = re.findall('Participant.*$', result_data, re.MULTILINE)  # Extracts only scores from text
        self._num_par = len(self._par)

        for i in range(self._num_par):
            self._par_data.append([])

        scores = re.findall(r': (.*)', result_data)     # Extracts only scores from text
        ind = -1                                        # Get an index ready to keep track of participant

        for i in range(len(scores)):

            if i % 3 == 0:                              # If benchmark score
                ind = ind + 1                           # Increase index value by 1
                temp = float(scores[i])                 # Convert score from string to decimal
                self._par_data[ind].append([temp])      # Save score to participant's list first index
            elif i % 3 == 1:                            # If channel rankings
                self._par_data[ind].append([])          # Save to 2nd index for participant
                for num in scores[i].split(', '): self._par_data[ind][1].append(float(num))  # Save rankings as decimals
            else:                                       # If channel reductions
                for num in scores[i].split(', '): self._par_data[ind][0].append(
                    float(num))                         # Add as decimals to first index


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

    def __getScore(self, gini=0):
        par_score = list()                              # Create a list to store score
        for par in self._par_data:                      # Parse through each participant
            par_score.append(par[gini])                 # Attach score
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

        return self.__getScore(gini=1)

    def delta(self, scores):
        all_diff = []
        diff = []
        for l in scores:
            for n in range(len(l) - 1):
                diff.append(l[n+1]-l[n])
            all_diff.append(diff)
        return all_diff

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

    def sortByParticipant(self, *args, gini=0):
        par_dic = _ParticipantDictionary.fromkeys(self._par, [])    # Create a participant dictionary
        scores = [self.__getScore(gini=gini)]

        for band in args:
            scores.append(band.__getScore(gini=gini))

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

    cwd = os.getcwd()  # Get the current working directory (cwd)

    all_data = "../Classification/Results/all_results.txt"  # Name of the file for all bands
    theta_data = "../Classification/Results/theta_results.txt"  # Name of the file for theta band
    delta_data = "../Classification/Results/delta_results.txt"  # Name of the file for delta band
    alpha_data = "../Classification/Results/alpha_results.txt"  # Name of the file for alpha band
    beta_data = "../Classification/Results/beta_results.txt"  # Name of the file for beta band
    gamma_data = "../Classification/Results/gamma_results.txt"  # Name of the file for theta band

    placement = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
                 'PO10']  # The position of each channel in order

    test2 = list(range(1, 33))

    all_bands = analysis(all_data, placement)  # Test correct inputs

    theta_band = analysis(theta_data, placement)
    delta_band = analysis(delta_data, placement)
    alpha_band = analysis(alpha_data, placement)
    beta_band = analysis(beta_data, placement)
    gamma_band = analysis(gamma_data, placement)

    sort = all_bands.sortByParticipant(theta_band, delta_band, alpha_band, beta_band, gamma_band, gini=0)

    print(sort)

    # acc = all_bands.accuracy()
    # print(acc)
    #
    # gini = all_bands.giniImportance()
    # print(gini)
    #
    # delta = all_bands.delta(acc)
    # print(delta)
    #
    # print(all_bands._num_chn)
    # avg = all_bands.average()
    #
    # print(avg)
    #
    # avg = all_bands.average(gini=1)
    #
    # print(avg)

    # ranked = all_bands.rankChannels()
    #
    # print(ranked)
