#######################################################################################################################
##
##  The analysis library contains methods to extract results from the classifier and provide graphical outputs of the
##  results.
##
#######################################################################################################################

import re


class analysis:

    def __init__(self, result_path, channel_key=None):

        if not isinstance(result_path, str):
            raise ValueError("Result path must be a string containing a valid file path.")

        self._path = result_path
        self._key = channel_key
        self._num_chn = len(self._key)
        self._par_data = []
        self._par = None
        self._num_par = 0
        self._readfile()

    def _readfile(self):
        result_file = open(self._path, 'r')
        result_data = result_file.read()
        self._par = re.findall('Participant.*$', result_data, re.MULTILINE)  # Extracts only scores from text
        print(self._par)
        self._num_par = len(self._par)

        for i in range(self._num_par):
            self._par_data.append([])

        scores = re.findall(r': (.*)', result_data)  # Extracts only scores from text

        ind = -1  # Get an index ready to keep track of participant

        for i in range(len(scores)):

            if i % 3 == 0:  # If benchmark score
                ind = ind + 1  # Increase index value by 1
                temp = float(scores[i])  # Convert score from string to decimal
                self._par_data[ind].append([temp])  # Save score to participant's list first index
            elif i % 3 == 1:  # If channel rankings
                self._par_data[ind].append([])  # Save to 2nd index for participant
                for num in scores[i].split(', '): self._par_data[ind][1].append(float(num))  # Save rankings as decimals
            else:  # If channel reductions
                for num in scores[i].split(', '): self._par_data[ind][0].append(
                    float(num))  # Add as decimals to first index

    def accuracy(self):
        for par in self._par_data:
            return par[0]

    def average(self):
        avg = []
        for c in range(self._num_chn):
            

        # for band in band_data:
        #     band_avg = []  # Create a list to store the average accuracy each reduction
        #     for channel in range(32):  # Go through all 32 channels
        #         avg_acc = 0  # Variable to store the average accuracy
        #         for i in range(12):  # Parse through the channel for all participants
        #             avg_acc = band[i][0][channel] + avg_acc  # Tally up the accuracies
        #
        #         avg_acc = avg_acc / 12  # Get the average accuracy
        #         band_avg.append(avg_acc)  # Append the average accuracy for the channel to the list
        #
        #     plt.plot(x_axis, band_avg, '-o', label=band_names[band_data.index(band)])  # Plot data
        #
        # plt.ylabel("Average Accuracy")  # Label the y-axis
        # plt.xlabel("Channels Used")  # Label the x-axis
        # plt.title("Average Accuracy Per Band")  # Add a title
        # plt.legend(loc=4)  # Add a legend to the plot
        # fname = "average_accuracy.png"  # Name for the plot savefile
        # plt.savefig(fname)  # Save the plot
        # plt.show()  # Create the plot
        #
        # del [band, band_avg, channel, avg_acc, i, fname]


if __name__ == "__main__":
    import os

    cwd = os.getcwd()  # Get the current working directory (cwd)
    all_data = "../Classification/Results/all_results.txt"  # Name of the file for all bands

    # all_bands = analysis(all_data, 1)                       # Test type identification
    # all_bands = analysis(all_data, [1, 2, 3])               # Test list validation
    placement = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
                 'PO10']  # The position of each channel in order

    all_bands = analysis(all_data, placement)  # Test correct inputs

    acc = all_bands.accuracy()

    print(acc)
