#################################################################################
#
#  This program is designed to parse through the results and determine which #
#  channels are selected for the most. It also calculates the average score for
#  each channel and assigns the 10-10 placement of the electrode.
#
#################################################################################

import os
import re
import matplotlib.pyplot as plt

###############################################################
#
# Load in the results from the classifier
#
###############################################################

cwd = os.getcwd()                                           # Get the current working directory (cwd)

all_data = "../Classification/Results/all_results.txt"      # Name of the file for all bands
theta_data = "../Classification/Results/theta_results.txt"  # Name of the file for theta band
delta_data = "../Classification/Results/delta_results.txt"  # Name of the file for delta band
alpha_data = "../Classification/Results/alpha_results.txt"  # Name of the file for alpha band
beta_data = "../Classification/Results/beta_results.txt"    # Name of the file for beta band
gamma_data = "../Classification/Results/gamma_results.txt"  # Name of the file for theta band

all_file = open(all_data, 'r')                              # Open text file for combined band data
theta_file = open(theta_data, 'r')                          # Open text file for theta band data
delta_file = open(delta_data, 'r')                          # Open text file for delta band data
alpha_file = open(alpha_data, 'r')                          # Open text file for alpha band data
beta_file = open(beta_data, 'r')                            # Open text file for beta band data
gamma_file = open(gamma_data, 'r')                          # Open text file for gamma band data

alld = all_file.read()                                      # Read the text file for combined data
theta = theta_file.read()                                   # Read the text file for thata data
delta = delta_file.read()                                   # Read the text file for delta data
alpha = alpha_file.read()                                   # Read the text file for alpha data
beta = beta_file.read()                                     # Read the text file for beta data
gamma = gamma_file.read()                                   # Read the text file for gamma data

band_options = [alld, theta, delta, alpha, beta, gamma]     # Save all results to one list for ease of indexing

del [all_data, theta_data, delta_data, alpha_data, beta_data, gamma_data, all_file, theta_file, delta_file, alpha_file,
     beta_file, gamma_file, alld, theta, delta, alpha, beta, gamma]

###############################################################
#
# Extract the numerical data from the text file
#
###############################################################

band_data = []  # Saves all participants data for a band

for band in band_options:

    par_data = [list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list(),
                list()]  # Creates a list with a list for each participant

    # Extract the participants data
    scores = re.findall(r': (.*)', band)  # Extracts only scores from text

    ind = -1  # Get an index ready to kwp track of participant

    for i in range(len(scores)):

        if i % 3 == 0:                      # If benchmark score
            ind = ind + 1                   # Increase index value by 1
            temp = float(scores[i])         # Convert score from string to decimal
            par_data[ind].append([temp])    # Save score to participant's list first index
        elif i % 3 == 1:                    # If channel rankings
            par_data[ind].append([])        # Save to 2nd index for participant
            for num in scores[i].split(', '): par_data[ind][1].append(float(num))  # Save rankings as decimals
        else:                               # If channel reductions
            for num in scores[i].split(', '): par_data[ind][0].append(float(num))  # Add as decimals to first index

    band_data.append(par_data)              # Save data to the band_data list

del [par_data, scores, ind, i, temp, num]

###############################################################
#
# Prepare lists for use in plot labelling
#
###############################################################

x_axis = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,
          3, 2, 1]  # Labels for the y-axis

band_names = ["All Bands", "Theta", "Delta", "Alpha", "Beta", "Gamma"]  # List for band name labels

par_names = ["Participant 1", "Participant 2", "Participant 3", "Participant 4", "Participant 5", "Participant 6",
             "Participant 7", "Participant 8", "Participant 9", "Participant 10", "Participant 11", "Participant 12"]
# List for participant name labels

placement = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
             'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2',
             'PO10']  # The position of each channel in order

###############################################################
#
# Plot the accuracy of each participant in every band
#
###############################################################

for band in band_data:                                                  # For each band
    plt.ylabel("Accuracy")                                              # Label the y-axis
    plt.xlabel("Channels Used")                                         # Label the x-axis
    for p in band:                                                      # For each participant in band
        plt.plot(x_axis, p[0], '-o', label=par_names[band.index(p)])    # Plot the accuracies per channels used

    plt.title(band_names[band_data.index(band)])                        # Add the band name as the title
    plt.legend(loc=4)                                                   # Add a legend to the plot
    fname = band_names[band_data.index(band)] + "_par_acc_per_chn.png"  # Name for the plot savefile
    plt.savefig(fname)                                                  # Save the plot
    plt.show()                                                          # Create plot

del [band, p, fname]

###############################################################
#
# Plot the accuracy of each band for each participant
#
###############################################################

for par in range(12):                               # For each participant
    plt.ylabel("Accuracy")                          # Label the y-axis
    plt.xlabel("Channels Used")                     # Label the x-axis
    for band in band_data:                          # Go through all bands
        plt.plot(x_axis, band[par][0], '-o', label=band_names[band_data.index(band)])   # Plot the accuracy per channel

    plt.title(par_names[par])                       # Add the participant as the title
    plt.legend(loc=4)                               # Add a legend to the plot
    fname = "P" + str(par+1) + "_acc_per_band.png"  # Name for the plot savefile
    plt.savefig(fname)                              # Save the plot
    plt.show()                                      # Create plot

del [par, band, fname]

###############################################################
#
# Plot the average accuracy over all participants per band
# against each other
#
###############################################################

for band in band_data:
    band_avg = []                                       # Create a list to store the average accuracy each reduction
    for channel in range(32):                           # Go through all 32 channels
        avg_acc = 0                                     # Variable to store the average accuracy
        for i in range(12):                             # Parse through the channel for all participants
            avg_acc = band[i][0][channel] + avg_acc     # Tally up the accuracies

        avg_acc = avg_acc/12                            # Get the average accuracy
        band_avg.append(avg_acc)                        # Append the average accuracy for the channel to the list

    plt.plot(x_axis, band_avg, '-o', label=band_names[band_data.index(band)])   # Plot data

plt.ylabel("Average Accuracy")                                  # Label the y-axis
plt.xlabel("Channels Used")                                     # Label the x-axis
plt.title("Average Accuracy Per Band")                          # Add a title
plt.legend(loc=4)                                               # Add a legend to the plot
fname ="average_accuracy.png"                                   # Name for the plot savefile
plt.savefig(fname)                                              # Save the plot
plt.show()                                                      # Create the plot

del [band, band_avg, channel, avg_acc, i, fname]

###############################################################
#
# Prepare lists for use in bar graphs and channel ranking
#
###############################################################

plt.rcParams['figure.figsize'] = [21, 14]   # Increase size of plot for clarity

plt.plot([1, 2], [2, 1], '-o')              # Dummy plot to set size changes into effect
plt.show()                                  # Show dummy plot

spacing = [x * 3 for x in range(0, 32)]     # Set spacing for bar plots

clrs = ['navy', 'orange', 'lawngreen', 'maroon', 'slateblue', 'darkmagenta', 'pink', 'magenta', 'yellowgreen', 'aqua',
        'dodgerblue', 'gold']               # Set the participant colours

avg_per = [[], [], [], [], [], []]          # Store the average gini importance per channel for each band

###############################################################
#
# Plot the gini importance of each channel for all participants
# in each band
#
# Note: This has been commented out due to poor visibility
#
###############################################################

# for band in band_data:
#     lct = -(3/12)*0.5
#     for p in band:
#         plt.bar([e + lct for e in spacing], p[1], width=0.2, label=par_names[band.index(p)],
#                 tick_label=placement, color=clrs[band.index(p)], align='edge')
#         lct = lct + (3/12)
#     plt.legend(loc=1, ncol=3)   # Add a legend to the plot
#     plt.show()

###############################################################
#
# Plot the average gini importance of each channel for all
# participants across each band
#
###############################################################

for band in band_data:                                                  # For each band
    chan_avg = []                                                       # List to store each channel's average per band
    for ch in range(32):                                                # For each channel
        avg_rank = 0                                                    # Zero out the average for each channel
        for p in band:                                                  # For each participant
            avg_rank = avg_rank + p[1][ch]                              # Tally the gini imp. over 12 participants

        chan_avg.append(avg_rank / 12)                                  # Store the average over 12 participants
        avg_per[band_data.index(band)].append(avg_rank)                 # Add the average performance to a list
        plt.text(x=spacing[ch], y=avg_rank/12+10, s=str(avg_rank/12))   # Write accuracies above each bar (in progress)

    plt.title(band_names[band_data.index(band)], fontsize=32)           # Set plot title
    plt.bar(spacing, chan_avg, width=2, tick_label=placement)           # Add bars to plot
    plt.yticks(fontsize=16)                                             # Increase y-axis label size
    plt.xticks(fontsize=16)                                             # Increase x-axis label size
    plt.ylabel("Average Gini Importance", fontsize=20)                  # Label the y-axis
    plt.xlabel("Channel Name", fontsize=20)                             # Label the x-axis
    fname = band_names[band_data.index(band)] + "_avg_gini.png"         # Name for the plot savefile
    plt.savefig(fname)                                                  # Save the plot
    plt.show()                                                          # Create the plot

del [band, chan_avg, ch, avg_rank, p, fname]

###############################################################
#
# Find the top ranking channels in each band in order of the
# highest average gini importance.
#
###############################################################

top_chn = [[], [], [], [], [], [], []]  # Store the top channel names in order

for band in band_data:                                                                          # For each band
    for ch in range(32):                                                                        # For each channel
        max_gini = avg_per[band_data.index(band)].index(max(avg_per[band_data.index(band)]))    # Index of highest gini
        top_chn[band_data.index(band)].append(placement[max_gini])                              # Store top channel
        avg_per[band_data.index(band)][max_gini] = 0.00                                         # Set top channel to 0

    print(top_chn[band_data.index(band)])                                                       # Print the top channels

del [band, ch, max_gini, avg_per]


