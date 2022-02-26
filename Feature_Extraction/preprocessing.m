%%  Clear workspace and console

clear;
clc;

%% Load in the raw EEG data and variables

Fs = 500;                                           % sampling rate

eeg_data = cell(12, 9);
eeg_spec_data = cell(12, 9);
N = zeros(12, 9);
T = zeros(12, 9);
f0 = zeros(12, 9);

for n = 1:1:12                                      % Get all 12 participants
    for h = 1:1:9                                   % Get session 1 through 9 for participant
        g = "HS_P" + n + "_S" + h +".csv";          % Make a csv file output name
        eeg_data{n, h} = readmatrix(g);             % Create a cell for each session
        eeg_spec_data{n, h} = fft(eeg_data{n, h});  % Spectrum data for EEG
        N(n, h) = length(eeg_data{n, h});           % number of samples per channel
        T(n, h) = N(n, h)/Fs;                       % The time axis for all data
        f0(n, h) = Fs/N(n, h);                      % frequency per sample
    end 
end 

ft = 1/Fs;                                          % Seconds per sample
    
clearvars n h g


%% Filter the eeg data to 0.1 - 50 Hz

cb = 0.2/(Fs/2);                                    % High-pass frequency adjusted for sampling rate
ca = 50/(Fs/2);                                     % Low-pass frequency adjusted to sampling rate
[b,a] = butter(4,[cb, ca]);                         % Create a Butterworth filter

clearvars cb ca 

%% Filter the EEG data 

eeg_filtered = cell(12, 9);
eeg_spec_filtered = cell(12, 9);

for n = 1:1:12
    for h = 1:1:9
        eeg_filtered{n, h} = filtfilt(b,a,eeg_data{n, h});    % Application of Butterworth filter
        eeg_spec_filtered{n, h} = fft(eeg_filtered{n, h});    % Convert to frequency domain 
    end 
end

clearvars a b  n h

%% Plot an instance of raw vs filtered EEG data

t = [ft:ft:T(1, 1)];                                % Create the time axis

figure(1)
subplot(4,1,1)

plot(t, eeg_data{1, 1}(:, 1)/10)
title('Time spectrum of the raw EEG signal')
xlabel('Time (s)'),ylabel('Amplitude (\muV)'),grid on

subplot(4,1,2)
plot(t, eeg_filtered{1, 1}(:, 1)/10)
title('Time spectrum of the filtered EEG signal')
xlabel('Time (s)'),ylabel('Amplitude (\muV)'),grid on

subplot(4,1,3)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_data{1, 1}(:, 1)))
title('Frequency spectrum of the raw EEG signal (0.1Hz a 60Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 60]);

subplot(4,1,4)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_filtered{1, 1}(:, 1)))
title('Frequency spectrum of the filtered EEG signal (0.1Hz a 60Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 60]);

clearvars eeg_data eeg_spec_data

%% Create filters for the five frequency bands 

hp = 0.2/(Fs/2); % High-pass frequency
lp = 4/(Fs/2);  % Low-pass frequency
[db,da] = butter(2,[hp, lp]);  % Delta band filter

hp = 4/(Fs/2); % High-pass frequency
lp = 8/(Fs/2);  % Low-pass frequency
[tb,ta] = butter(4,[hp, lp]);  % Theta band filter

hp = 8/(Fs/2); % High-pass frequency
lp = 12/(Fs/2);  % Low-pass frequency
[ab,aa] = butter(4,[hp, lp]);  % Alpha band filter

hp = 12/(Fs/2); % High-pass frequency
lp = 26/(Fs/2);  % Low-pass frequency
[bb,ba] = butter(4,[hp, lp]);  % Beta band filter

hp = 26/(Fs/2); % High-pass frequency
lp = 50/(Fs/2);  % Low-pass frequency
[gb,ga] = butter(4,[hp, lp]);  % Gamma band filter

clearvars lp hp

%% Apply frequency band filter on EEG data

eeg_delta = cell(12, 9);
eeg_theta = cell(12, 9);
eeg_alpha = cell(12, 9);
eeg_beta = cell(12, 9);
eeg_gamma = cell(12, 9);

for n = 1:1:12
    for h = 1:1:9
        eeg_delta{n, h} = filtfilt(db,da,eeg_filtered{n, h});  % Extract delta band
        eeg_theta{n, h} = filtfilt(tb,ta,eeg_filtered{n, h});  % Extract theta band
        eeg_alpha{n, h} = filtfilt(ab,aa,eeg_filtered{n, h});  % Extract alpha band
        eeg_beta{n, h}  = filtfilt(bb,ba,eeg_filtered{n, h});  % Extract beta band
        eeg_gamma{n, h} = filtfilt(gb,ga,eeg_filtered{n, h});  % Extract gamma band
    end 
end

clearvars n h db da ta tb aa ab bb ba ga gb

%% Plot an example of each wave 

figure(2)

subplot(5,1,1)
plot(t, eeg_delta{1, 1}(:, 1)/10)
title('Delta Band')
xlabel('Time (ms)'),ylabel('Amplitude (\muV)'),grid on
xlim([0 10]);

subplot(5,1,2)
plot(t, eeg_theta{1, 1}(:, 1)/10)
title('Theta Band')
xlabel('Time (s)'),ylabel('Amplitude (\muV)'),grid on
xlim([0 10]);

subplot(5,1,3)
plot(t, eeg_alpha{1, 1}(:, 1)/10)
title('Alpha Band')
xlabel('Time (s)'),ylabel('Amplitude (\muV)'),grid on
xlim([0 10]);

subplot(5,1,4)
plot(t, eeg_beta{1, 1}(:, 1)/10)
title('Beta Band')
xlabel('Time (s)'),ylabel('Amplitude (\muV)'),grid on
xlim([0 10]);

subplot(5,1,5)
plot(t, eeg_gamma{1, 1}(:, 1)/10)
title('Gamma Band')
xlabel('Time (s)'),ylabel('Amplitude (\muV)'),grid on
xlim([0 10]);

%% Get frequency band spectral transform for delta and theta

eeg_spec_delta = cell(12, 9);
eeg_spec_theta = cell(12, 9);
eeg_spec_alpha = cell(12, 9);
eeg_spec_beta = cell(12, 9);
eeg_spec_gamma = cell(12, 9);

for n = 1:1:12
    for h = 1:1:9
        eeg_spec_delta{n, h} = fft(eeg_delta{n, h});       % Spectral delta band
        eeg_spec_theta{n, h} = fft(eeg_theta{n, h});       % Spectral theta band
    end
end

clearvars n h

%% Get the transform for alpha and beta

for n = 1:1:12
    for h = 1:1:9
        eeg_spec_alpha{n, h} = fft(eeg_alpha{n, h});       % Spectral alpha band
        eeg_spec_beta{n, h} = fft(eeg_beta{n, h});         % Spectral beta band
    end
end

clearvars n h

%% Get the transform for gamma

for n = 1:1:12
    for h = 1:1:9
        eeg_spec_gamma{n, h} = fft(eeg_gamma{n, h});       % Spectral gamma band
    end
end

clearvars n h

%% Plot spectral data of each band

subplot(5,1,1)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_delta{1, 1}(:, 1)))
title('Frequency spectrum of the delta band (0.1Hz a 50Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 50]);

subplot(5,1,2)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_theta{1, 1}(:, 1)))
title('Frequency spectrum of the theta band (0.1Hz a 50Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 50]);

subplot(5,1,3)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_alpha{1, 1}(:, 1)))
title('Frequency spectrum of the alpha band (0.1Hz a 50Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 50]);

subplot(5,1,4)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_beta{1, 1}(:, 1)))
title('Frequency spectrum of the beta band (0.1Hz a 50Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 50]);

subplot(5,1,5)
plot(f0(1, 1)*(0:N(1, 1)-1),abs(eeg_spec_gamma{1, 1}(:, 1)))
title('Frequency spectrum of the gamma (0.1Hz a 50Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy (\muV)'),grid on
xlim([0 50]);