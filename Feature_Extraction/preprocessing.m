%%  Clear workspace and console

clear;
clc;

%% Load in the raw EEG data and variables

p = 1;                                          % Choose the participant

Fs = 500;                                       % sampling rate

eeg_data = cell(1, 9);
eeg_spec_data = cell(1, 9);
N = zeros(1, 9);
T = zeros(1, 9);
f0 = zeros(1, 9);

for h = 1:1:9                                   % Get session 1 through 9 for participant
    g = "P" + p + "\HS_P" + p + "_S" + h +".csv";          % Make a csv file input name
    eeg_data{1, h} = readmatrix(g);             % Create a cell for each session
    eeg_spec_data{1, h} = fft(eeg_data{1, h});  % Spectrum data for EEG
    N(1, h) = length(eeg_data{1, h});           % number of samples per channel
    T(1, h) = N(1, h)/Fs;                       % The time axis for all data
    f0(1, h) = Fs/N(1, h);                      % frequency per sample
end

ft = 1/Fs;                                      % Seconds per sample

clearvars h g

%% Filter the eeg data to 0.1 - 50 Hz

cb = 0.2/(Fs/2);                                    % High-pass frequency adjusted for sampling rate
ca = 50/(Fs/2);                                     % Low-pass frequency adjusted to sampling rate
[b,a] = butter(4,[cb, ca]);                         % Create a Butterworth filter

clearvars cb ca

%% Filter the EEG data

eeg_filtered = cell(1, 9);
eeg_spec_filtered = cell(1, 9);


for h = 1:1:9
    eeg_filtered{1, h} = filtfilt(b,a,eeg_data{1, h});    % Application of Butterworth filter
    eeg_spec_filtered{1, h} = fft(eeg_filtered{1, h});    % Convert to frequency domain
end


clearvars a b h

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

hp = 0.2/(Fs/2);                % High-pass frequency
lp = 4/(Fs/2);                  % Low-pass frequency
[db,da] = butter(2,[hp, lp]);   % Delta band filter

hp = 4/(Fs/2);                  % High-pass frequency
lp = 8/(Fs/2);                  % Low-pass frequency
[tb,ta] = butter(4,[hp, lp]);   % Theta band filter

hp = 8/(Fs/2);                  % High-pass frequency
lp = 12/(Fs/2);                 % Low-pass frequency
[ab,aa] = butter(4,[hp, lp]);   % Alpha band filter

hp = 12/(Fs/2);                 % High-pass frequency
lp = 26/(Fs/2);                 % Low-pass frequency
[bb,ba] = butter(4,[hp, lp]);   % Beta band filter

hp = 26/(Fs/2);                 % High-pass frequency
lp = 50/(Fs/2);                 % Low-pass frequency
[gb,ga] = butter(4,[hp, lp]);   % Gamma band filter

clearvars lp hp

%% Apply frequency band filter on EEG data

eeg_delta = cell(1, 9);
eeg_theta = cell(1, 9);
eeg_alpha = cell(1, 9);
eeg_beta = cell(1, 9);
eeg_gamma = cell(1, 9);


for h = 1:1:9
    eeg_delta{1, h} = filtfilt(db,da,eeg_filtered{1, h});  % Extract delta band
    eeg_theta{1, h} = filtfilt(tb,ta,eeg_filtered{1, h});  % Extract theta band
    eeg_alpha{1, h} = filtfilt(ab,aa,eeg_filtered{1, h});  % Extract alpha band
    eeg_beta{1, h}  = filtfilt(bb,ba,eeg_filtered{1, h});  % Extract beta band
    eeg_gamma{1, h} = filtfilt(gb,ga,eeg_filtered{1, h});  % Extract gamma band
end


clearvars h db da ta tb aa ab bb ba ga gb

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

eeg_spec_delta = cell(1, 9);
eeg_spec_theta = cell(1, 9);
eeg_spec_alpha = cell(1, 9);
eeg_spec_beta = cell(1, 9);
eeg_spec_gamma = cell(1, 9);

for h = 1:1:9
    eeg_spec_delta{1, h} = fft(eeg_delta{1, h});       % Spectral delta band
    eeg_spec_theta{1, h} = fft(eeg_theta{1, h});       % Spectral theta band
    eeg_spec_alpha{1, h} = fft(eeg_alpha{1, h});       % Spectral alpha band
    eeg_spec_beta{1, h} = fft(eeg_beta{1, h});         % Spectral beta band
    eeg_spec_gamma{1, h} = fft(eeg_gamma{1, h});       % Spectral gamma band
end

clearvars h

%% Plot spectral data of each band

figure(3)

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

clearvars f0 ft T t

%% Windowing
%
%  The windowing section will create 0.5s segments of the EEG data for each
%  channel, with a 50% overlap. 

w_size = Fs * 0.5;          % The number of samples per 0.5s segment
overlap = w_size/2;         % Create a 50% overlap

n_win =floor(N/125 - 1);    % Number of windows per channel

clearvars Fs N

%% Feature Extraction
%
%  This section of the code is used to extract features from the EEG data.
%  The features extracated in this program are: Mean, Standard Deviation, 
%  Mean Absolute Value, Root Mean Square, Skewness, Kurtosis, Hjorth
%  Activity, Hjorth Mobility, Hjorth Complexity, Shannon's Entropy, 
%  Spectral Entropy, and Power Spectrum Density for the overall EEG data, 
%  and for each frequency band.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note that the arrays are calculated per window for all 32 channels

eeg_avg = cell(1, 9);
eeg_std = cell(1, 9);
eeg_mav = cell(1, 9);
eeg_rms = cell(1, 9);
eeg_skw = cell(1, 9);
eeg_kur = cell(1, 9);
eeg_hac = cell(1, 9);
eeg_hmb = cell(1, 9);
eeg_hcp = cell(1, 9);
eeg_she = cell(1, 9);
eeg_spe = cell(1, 9);
eeg_psd = cell(1, 9);

for h=1:1:9    
    eeg_avg{1, h} = zeros(n_win(1, h),32); % Average
    eeg_std{1, h} = zeros(n_win(1, h),32); % Standard Deviation
    eeg_mav{1, h} = zeros(n_win(1, h),32); % Mean Absolute Value
    eeg_rms{1, h} = zeros(n_win(1, h),32); % Root Mean Square
    eeg_skw{1, h} = zeros(n_win(1, h),32); % Skewness
    eeg_kur{1, h} = zeros(n_win(1, h),32); % Kurtosis
    eeg_hac{1, h} = zeros(n_win(1, h),32); % Hjorth Activity
    eeg_hmb{1, h} = zeros(n_win(1, h),32); % Hjorth Mobility
    eeg_hcp{1, h} = zeros(n_win(1, h),32); % Hjorth Complexity
    eeg_she{1, h} = zeros(n_win(1, h),32); % Shannon's EntropyW
    eeg_spe{1, h} = zeros(n_win(1, h),32); % Spectral Entropy
    eeg_psd{1, h} = zeros(n_win(1, h),32); % Power Spectral Density
end 

for h = 1:1:9
    w_start = 1;                % Start at the first sample
    w_end = w_size;             % End at window length
    for n = 1:1:n_win(1, h)
        eeg_avg{1, h}(n, :) = mean(eeg_filtered{1, h}(w_start:w_end, :));                       % Average for window
        eeg_std{1, h}(n, :) = std(eeg_filtered{1, h}(w_start:w_end, :));                        % Standard Deviation for window
        eeg_mav{1, h}(n, :) = mean(abs(eeg_filtered{1, h}(w_start:w_end, :)));                  % Mean Absolute Value for window
        eeg_rms{1, h}(n, :) = rms(eeg_filtered{1, h}(w_start:w_end, :));                        % Root Mean Square for window
        eeg_skw{1, h}(n, :) = skewness(eeg_filtered{1, h}(w_start:w_end, :));                   % Skewness for window
        eeg_kur{1, h}(n, :) = kurtosis(eeg_filtered{1, h}(w_start:w_end, :));                   % Kurtosis for window
        eeg_hac{1, h}(n, :) = var(eeg_filtered{1, h}(w_start:w_end, :));                        % Hjorth Activity for window
        [eeg_hmb{1, h}(n, :), eeg_hcp{1, h}(n, :)] = ...
            HjorthParameters(eeg_filtered{1, h}(w_start:w_end, :));                             % Hjorth Mobility and Complexity for window
        eeg_she{1, h}(n, :) = wentropy(eeg_filtered{1, h}(w_start:w_end, :), 'shannon');        % Shannon's Entropy for window
        eeg_spe{1, h}(n, :) = wentropy(eeg_spec_filtered{1, h}(w_start:w_end, :), 'shannon');   % Spectral Entropy for window
        eeg_psd{1, h}(n, :) = mean(pburg(eeg_filtered{1, h}(w_start:w_end, :), 4));             % Power Spectral Density for window
    
        w_start = w_start + overlap;        % Shift the start of the next window forward by 50%
        w_end = w_start + w_size;           % Shift end by 250 samples
    end
end

feats = {eeg_avg, eeg_std, eeg_mav, eeg_rms, eeg_skw, eeg_kur, eeg_hac, ...
    eeg_hmb, eeg_hcp, eeg_she, eeg_spe, eeg_psd};

clearvars h n eeg_avg eeg_std eeg_mav eeg_rms eeg_skw eeg_kur eeg_hac ...
    eeg_hmb eeg_hcp eeg_she eeg_spe eeg_psd eeg_filtered eeg_spec_filtered

%% Export features for overall EEG data

table_names = ["Participant", "Feature", "ch1", "ch2", "ch3", "ch4", "ch5", ...
    "ch6",  "ch7", "ch8", "ch9", "ch10", "ch11", "ch12", "ch13", "ch14",...
    "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", "ch21", "ch22", ...
    "ch23", "ch24", "ch25", "ch26", "ch27", "ch28", "ch29", "ch30", ...
    "ch31", "ch32"];

var_types = ["double", "string", "double", "double", "double", "double", "double",... 
    "double", "double", "double", "double", "double", "double", "double",...
    "double", "double", "double", "double", "double", "double", "double",...
    "double", "double", "double", "double", "double", "double", "double",...
    "double", "double", "double", "double", "double", "double"];

labels = ["Mean", "Standard Deviation", "Mean Absolute Value", ...
    "Root Mean Square", "Skewness", "Kurtosis", "Hjorth Activity", ...
    "Hjorth Mobility", "Hjorth Complexity", "Shannon's Entropy", ...
    "Spectral Entropy", "Power Spectral Density"];

for h = 1:1:9
    eeg_features{1, h} = table('Size', [n_win(1, h)*12 ,34], ...
        'VariableTypes', var_types);                                                % Create table to store all features
    allVars = 1:width(eeg_features{1, h});
    eeg_features{1, h} = renamevars(eeg_features{1, h}, allVars, table_names);      % Column names
    eeg_features{1, h}(:, 1) = {p};

    for n = 1:1:12
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 2) = {labels(1, n)};  % Label features on table
        feat = array2table(real(feats{1, n}{1, h}));
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 3:34) = feat;         % Label features on table
    end
    g = "P" + p + "\Features_P" + p + "_S" + h +".csv";                                        % Make a csv file output name
    writetable(eeg_features{1, h}, g);
end 

clearvars h allVars eeg_features feat feats n g

%% Repeat feature extraction for delta band

delta_avg = cell(1, 9);
delta_std = cell(1, 9);
delta_mav = cell(1, 9);
delta_rms = cell(1, 9);
delta_skw = cell(1, 9);
delta_kur = cell(1, 9);
delta_hac = cell(1, 9);
delta_hmb = cell(1, 9);
delta_hcp = cell(1, 9);
delta_she = cell(1, 9);
delta_spe = cell(1, 9);
delta_psd = cell(1, 9);

for h=1:1:9    
    delta_avg{1, h} = zeros(n_win(1, h),32); % Average
    delta_std{1, h} = zeros(n_win(1, h),32); % Standard Deviation
    delta_mav{1, h} = zeros(n_win(1, h),32); % Mean Absolute Value
    delta_rms{1, h} = zeros(n_win(1, h),32); % Root Mean Square
    delta_skw{1, h} = zeros(n_win(1, h),32); % Skewness
    delta_kur{1, h} = zeros(n_win(1, h),32); % Kurtosis
    delta_hac{1, h} = zeros(n_win(1, h),32); % Hjorth Activity
    delta_hmb{1, h} = zeros(n_win(1, h),32); % Hjorth Mobility
    delta_hcp{1, h} = zeros(n_win(1, h),32); % Hjorth Complexity
    delta_she{1, h} = zeros(n_win(1, h),32); % Shannon's EntropyW
    delta_spe{1, h} = zeros(n_win(1, h),32); % Spectral Entropy
    delta_psd{1, h} = zeros(n_win(1, h),32); % Power Spectral Density
end 

for h = 1:1:9
    w_start = 1;                % Start at the first sample
    w_end = w_size;             % End at window length
    for n = 1:1:n_win(1, h)
        delta_avg{1, h}(n, :) = mean(eeg_delta{1, h}(w_start:w_end, :));                   % Average for window
        delta_std{1, h}(n, :) = std(eeg_delta{1, h}(w_start:w_end, :));                    % Standard Deviation for window
        delta_mav{1, h}(n, :) = mean(abs(eeg_delta{1, h}(w_start:w_end, :)));              % Mean Absolute Value for window
        delta_rms{1, h}(n, :) = rms(eeg_delta{1, h}(w_start:w_end, :));                    % Root Mean Square for window
        delta_skw{1, h}(n, :) = skewness(eeg_delta{1, h}(w_start:w_end, :));               % Skewness for window
        delta_kur{1, h}(n, :) = kurtosis(eeg_delta{1, h}(w_start:w_end, :));               % Kurtosis for window
        delta_hac{1, h}(n, :) = var(eeg_delta{1, h}(w_start:w_end, :));                    % Hjorth Activity for window
        [delta_hmb{1, h}(n, :), delta_hcp{1, h}(n, :)] = ...
            HjorthParameters(eeg_delta{1, h}(w_start:w_end, :));                           % Hjorth Mobility and Complexity for window
        delta_she{1, h}(n, :) = wentropy(eeg_delta{1, h}(w_start:w_end, :), 'shannon');    % Shannon's Entropy for window
        delta_spe{1, h}(n, :) = wentropy(eeg_spec_delta{1, h}(w_start:w_end, :), 'shannon');     % Spectral Entropy for window
        delta_psd{1, h}(n, :) = mean(pburg(eeg_delta{1, h}(w_start:w_end, :), 4));               % Power Spectral Density for window
    
        w_start = w_start + overlap;
        w_end = w_start + w_size;
    end
end

feats = {delta_avg, delta_std, delta_mav, delta_rms, delta_skw, delta_kur, delta_hac, ...
    delta_hmb, delta_hcp, delta_she, delta_spe, delta_psd};

clearvars h n delta_avg delta_std delta_mav delta_rms delta_skw delta_kur delta_hac ...
    delta_hmb delta_hcp delta_she delta_spe delta_psd eeg_delta eeg_spec_delta

%% Export features for the delta band

for h = 1:1:9
    eeg_features{1, h} = table('Size', [n_win(1, h)*12 ,34], ...
        'VariableTypes', var_types);                                                % Create table to store all features
    allVars = 1:width(eeg_features{1, h});
    eeg_features{1, h} = renamevars(eeg_features{1, h}, allVars, table_names);      % Column names
    eeg_features{1, h}(:, 1) = {p};

    for n = 1:1:12
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 2) = {labels(1, n)};  % Label features on table
        feat = array2table(real(feats{1, n}{1, h}));
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 3:34) = feat;         % Label features on table
    end
    g = "P" + p + "\Features_P" + p + "_S" + h +"_Delta.csv";                                  % Make a csv file output name
    writetable(eeg_features{1, h}, g);
end 

clearvars h allVars eeg_features feat feats n g

%% Repeat feature extraction for theta band

theta_avg = cell(1, 9);
theta_std = cell(1, 9);
theta_mav = cell(1, 9);
theta_rms = cell(1, 9);
theta_skw = cell(1, 9);
theta_kur = cell(1, 9);
theta_hac = cell(1, 9);
theta_hmb = cell(1, 9);
theta_hcp = cell(1, 9);
theta_she = cell(1, 9);
theta_spe = cell(1, 9);
theta_psd = cell(1, 9);

for h=1:1:9    
    theta_avg{1, h} = zeros(n_win(1, h),32); % Average
    theta_std{1, h} = zeros(n_win(1, h),32); % Standard Deviation
    theta_mav{1, h} = zeros(n_win(1, h),32); % Mean Absolute Value
    theta_rms{1, h} = zeros(n_win(1, h),32); % Root Mean Square
    theta_skw{1, h} = zeros(n_win(1, h),32); % Skewness
    theta_kur{1, h} = zeros(n_win(1, h),32); % Kurtosis
    theta_hac{1, h} = zeros(n_win(1, h),32); % Hjorth Activity
    theta_hmb{1, h} = zeros(n_win(1, h),32); % Hjorth Mobility
    theta_hcp{1, h} = zeros(n_win(1, h),32); % Hjorth Complexity
    theta_she{1, h} = zeros(n_win(1, h),32); % Shannon's EntropyW
    theta_spe{1, h} = zeros(n_win(1, h),32); % Spectral Entropy
    theta_psd{1, h} = zeros(n_win(1, h),32); % Power Spectral Density
end 

for h = 1:1:9
    w_start = 1;                % Start at the first sample
    w_end = w_size;             % End at window length
    for n = 1:1:n_win(1, h)
        theta_avg{1, h}(n, :) = mean(eeg_theta{1, h}(w_start:w_end, :));                   % Average for window
        theta_std{1, h}(n, :) = std(eeg_theta{1, h}(w_start:w_end, :));                    % Standard Deviation for window
        theta_mav{1, h}(n, :) = mean(abs(eeg_theta{1, h}(w_start:w_end, :)));              % Mean Absolute Value for window
        theta_rms{1, h}(n, :) = rms(eeg_theta{1, h}(w_start:w_end, :));                    % Root Mean Square for window
        theta_skw{1, h}(n, :) = skewness(eeg_theta{1, h}(w_start:w_end, :));               % Skewness for window
        theta_kur{1, h}(n, :) = kurtosis(eeg_theta{1, h}(w_start:w_end, :));               % Kurtosis for window
        theta_hac{1, h}(n, :) = var(eeg_theta{1, h}(w_start:w_end, :));                    % Hjorth Activity for window
        [theta_hmb{1, h}(n, :), theta_hcp{1, h}(n, :)] = ...
            HjorthParameters(eeg_theta{1, h}(w_start:w_end, :));                         % Hjorth Mobility and Complexity for window
        theta_she{1, h}(n, :) = wentropy(eeg_theta{1, h}(w_start:w_end, :), 'shannon');    % Shannon's Entropy for window
        theta_spe{1, h}(n, :) = wentropy(eeg_spec_theta{1, h}(w_start:w_end, :), 'shannon');     % Spectral Entropy for window
        theta_psd{1, h}(n, :) = mean(pburg(eeg_theta{1, h}(w_start:w_end, :), 4));               % Power Spectral Density for window
    
        w_start = w_start + overlap;
        w_end = w_start + w_size;
    end
end

feats = {theta_avg, theta_std, theta_mav, theta_rms, theta_skw, theta_kur, theta_hac, ...
    theta_hmb, theta_hcp, theta_she, theta_spe, theta_psd};

clearvars h n theta_avg theta_std theta_mav theta_rms theta_skw theta_kur theta_hac ...
    theta_hmb theta_hcp theta_she theta_spe theta_psd eeg_theta eeg_spec_theta

%% Export features for the theta band

for h = 1:1:9
    eeg_features{1, h} = table('Size', [n_win(1, h)*12 ,34], ...
        'VariableTypes', var_types);                                                % Create table to store all features
    allVars = 1:width(eeg_features{1, h});
    eeg_features{1, h} = renamevars(eeg_features{1, h}, allVars, table_names);      % Column names
    eeg_features{1, h}(:, 1) = {p};

    for n = 1:1:12
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 2) = {labels(1, n)};  % Label features on table
        feat = array2table(real(feats{1, n}{1, h}));
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 3:34) = feat;         % Label features on table
    end
    g = "P" + p + "\Features_P" + p + "_S" + h +"_Theta.csv";                                  % Make a csv file output name
    writetable(eeg_features{1, h}, g);
end 

clearvars h allVars eeg_features feat feats n g

%% Repeat feature extraction for alpha band

alpha_avg = cell(1, 9);
alpha_std = cell(1, 9);
alpha_mav = cell(1, 9);
alpha_rms = cell(1, 9);
alpha_skw = cell(1, 9);
alpha_kur = cell(1, 9);
alpha_hac = cell(1, 9);
alpha_hmb = cell(1, 9);
alpha_hcp = cell(1, 9);
alpha_she = cell(1, 9);
alpha_spe = cell(1, 9);
alpha_psd = cell(1, 9);

for h=1:1:9    
    alpha_avg{1, h} = zeros(n_win(1, h),32); % Average
    alpha_std{1, h} = zeros(n_win(1, h),32); % Standard Deviation
    alpha_mav{1, h} = zeros(n_win(1, h),32); % Mean Absolute Value
    alpha_rms{1, h} = zeros(n_win(1, h),32); % Root Mean Square
    alpha_skw{1, h} = zeros(n_win(1, h),32); % Skewness
    alpha_kur{1, h} = zeros(n_win(1, h),32); % Kurtosis
    alpha_hac{1, h} = zeros(n_win(1, h),32); % Hjorth Activity
    alpha_hmb{1, h} = zeros(n_win(1, h),32); % Hjorth Mobility
    alpha_hcp{1, h} = zeros(n_win(1, h),32); % Hjorth Complexity
    alpha_she{1, h} = zeros(n_win(1, h),32); % Shannon's EntropyW
    alpha_spe{1, h} = zeros(n_win(1, h),32); % Spectral Entropy
    alpha_psd{1, h} = zeros(n_win(1, h),32); % Power Spectral Density
end 

for h = 1:1:9
    w_start = 1;                % Start at the first sample
    w_end = w_size;             % End at window length
    for n = 1:1:n_win(1, h)
        alpha_avg{1, h}(n, :) = mean(eeg_alpha{1, h}(w_start:w_end, :));                   % Average for window
        alpha_std{1, h}(n, :) = std(eeg_alpha{1, h}(w_start:w_end, :));                    % Standard Deviation for window
        alpha_mav{1, h}(n, :) = mean(abs(eeg_alpha{1, h}(w_start:w_end, :)));              % Mean Absolute Value for window
        alpha_rms{1, h}(n, :) = rms(eeg_alpha{1, h}(w_start:w_end, :));                    % Root Mean Square for window
        alpha_skw{1, h}(n, :) = skewness(eeg_alpha{1, h}(w_start:w_end, :));               % Skewness for window
        alpha_kur{1, h}(n, :) = kurtosis(eeg_alpha{1, h}(w_start:w_end, :));               % Kurtosis for window
        alpha_hac{1, h}(n, :) = var(eeg_alpha{1, h}(w_start:w_end, :));                    % Hjorth Activity for window
        [alpha_hmb{1, h}(n, :), alpha_hcp{1, h}(n, :)] = ...
            HjorthParameters(eeg_alpha{1, h}(w_start:w_end, :));                         % Hjorth Mobility and Complexity for window
        alpha_she{1, h}(n, :) = wentropy(eeg_alpha{1, h}(w_start:w_end, :), 'shannon');    % Shannon's Entropy for window
        alpha_spe{1, h}(n, :) = wentropy(eeg_spec_alpha{1, h}(w_start:w_end, :), 'shannon');     % Spectral Entropy for window
        alpha_psd{1, h}(n, :) = mean(pburg(eeg_alpha{1, h}(w_start:w_end, :), 4));               % Power Spectral Density for window
    
        w_start = w_start + overlap;
        w_end = w_start + w_size;
    end
end

feats = {alpha_avg, alpha_std, alpha_mav, alpha_rms, alpha_skw, alpha_kur, alpha_hac, ...
    alpha_hmb, alpha_hcp, alpha_she, alpha_spe, alpha_psd};

clearvars h n alpha_avg alpha_std alpha_mav alpha_rms alpha_skw alpha_kur alpha_hac ...
    alpha_hmb alpha_hcp alpha_she alpha_spe alpha_psd eeg_alpha eeg_spec_alpha

%% Export features for the alpha band

for h = 1:1:9
    eeg_features{1, h} = table('Size', [n_win(1, h)*12 ,34], ...
        'VariableTypes', var_types);                                                % Create table to store all features
    allVars = 1:width(eeg_features{1, h});
    eeg_features{1, h} = renamevars(eeg_features{1, h}, allVars, table_names);      % Column names
    eeg_features{1, h}(:, 1) = {p};

    for n = 1:1:12
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 2) = {labels(1, n)};  % Label features on table
        feat = array2table(real(feats{1, n}{1, h}));
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 3:34) = feat;         % Label features on table
    end
    g = "P" + p + "\Features_P" + p + "_S" + h +"_Alpha.csv";                                  % Make a csv file output name
    writetable(eeg_features{1, h}, g);
end 

clearvars h allVars eeg_features feat feats n g

%% Repeat feature extraction for beta band

beta_avg = cell(1, 9);
beta_std = cell(1, 9);
beta_mav = cell(1, 9);
beta_rms = cell(1, 9);
beta_skw = cell(1, 9);
beta_kur = cell(1, 9);
beta_hac = cell(1, 9);
beta_hmb = cell(1, 9);
beta_hcp = cell(1, 9);
beta_she = cell(1, 9);
beta_spe = cell(1, 9);
beta_psd = cell(1, 9);

for h=1:1:9    
    beta_avg{1, h} = zeros(n_win(1, h),32); % Average
    beta_std{1, h} = zeros(n_win(1, h),32); % Standard Deviation
    beta_mav{1, h} = zeros(n_win(1, h),32); % Mean Absolute Value
    beta_rms{1, h} = zeros(n_win(1, h),32); % Root Mean Square
    beta_skw{1, h} = zeros(n_win(1, h),32); % Skewness
    beta_kur{1, h} = zeros(n_win(1, h),32); % Kurtosis
    beta_hac{1, h} = zeros(n_win(1, h),32); % Hjorth Activity
    beta_hmb{1, h} = zeros(n_win(1, h),32); % Hjorth Mobility
    beta_hcp{1, h} = zeros(n_win(1, h),32); % Hjorth Complexity
    beta_she{1, h} = zeros(n_win(1, h),32); % Shannon's EntropyW
    beta_spe{1, h} = zeros(n_win(1, h),32); % Spectral Entropy
    beta_psd{1, h} = zeros(n_win(1, h),32); % Power Spectral Density
end 

for h = 1:1:9
    w_start = 1;                % Start at the first sample
    w_end = w_size;             % End at window length
    for n = 1:1:n_win(1, h)
        beta_avg{1, h}(n, :) = mean(eeg_beta{1, h}(w_start:w_end, :));                   % Average for window
        beta_std{1, h}(n, :) = std(eeg_beta{1, h}(w_start:w_end, :));                    % Standard Deviation for window
        beta_mav{1, h}(n, :) = mean(abs(eeg_beta{1, h}(w_start:w_end, :)));              % Mean Absolute Value for window
        beta_rms{1, h}(n, :) = rms(eeg_beta{1, h}(w_start:w_end, :));                    % Root Mean Square for window
        beta_skw{1, h}(n, :) = skewness(eeg_beta{1, h}(w_start:w_end, :));               % Skewness for window
        beta_kur{1, h}(n, :) = kurtosis(eeg_beta{1, h}(w_start:w_end, :));               % Kurtosis for window
        beta_hac{1, h}(n, :) = var(eeg_beta{1, h}(w_start:w_end, :));                    % Hjorth Activity for window
        [beta_hmb{1, h}(n, :), beta_hcp{1, h}(n, :)] = ...
            HjorthParameters(eeg_beta{1, h}(w_start:w_end, :));                         % Hjorth Mobility and Complexity for window
        beta_she{1, h}(n, :) = wentropy(eeg_beta{1, h}(w_start:w_end, :), 'shannon');    % Shannon's Entropy for window
        beta_spe{1, h}(n, :) = wentropy(eeg_spec_beta{1, h}(w_start:w_end, :), 'shannon');     % Spectral Entropy for window
        beta_psd{1, h}(n, :) = mean(pburg(eeg_beta{1, h}(w_start:w_end, :), 4));               % Power Spectral Density for window
    
        w_start = w_start + overlap;
        w_end = w_start + w_size;
    end
end

feats = {beta_avg, beta_std, beta_mav, beta_rms, beta_skw, beta_kur, beta_hac, ...
    beta_hmb, beta_hcp, beta_she, beta_spe, beta_psd};

clearvars h n beta_avg beta_std beta_mav beta_rms beta_skw beta_kur beta_hac ...
    beta_hmb beta_hcp beta_she beta_spe beta_psd eeg_beta eeg_spec_beta

%% Export features for the beta band

for h = 1:1:9
    eeg_features{1, h} = table('Size', [n_win(1, h)*12 ,34], ...
        'VariableTypes', var_types);                                                % Create table to store all features
    allVars = 1:width(eeg_features{1, h});
    eeg_features{1, h} = renamevars(eeg_features{1, h}, allVars, table_names);      % Column names
    eeg_features{1, h}(:, 1) = {p};

    for n = 1:1:12
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 2) = {labels(1, n)};  % Label features on table
        feat = array2table(real(feats{1, n}{1, h}));
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 3:34) = feat;         % Label features on table
    end
    g = "P" + p + "\Features_P" + p + "_S" + h +"_Beta.csv";                                  % Make a csv file output name
    writetable(eeg_features{1, h}, g);
end 

clearvars h allVars eeg_features feat feats n g

%% Repeat feature extraction for gamma band

gamma_avg = cell(1, 9);
gamma_std = cell(1, 9);
gamma_mav = cell(1, 9);
gamma_rms = cell(1, 9);
gamma_skw = cell(1, 9);
gamma_kur = cell(1, 9);
gamma_hac = cell(1, 9);
gamma_hmb = cell(1, 9);
gamma_hcp = cell(1, 9);
gamma_she = cell(1, 9);
gamma_spe = cell(1, 9);
gamma_psd = cell(1, 9);

for h=1:1:9    
    gamma_avg{1, h} = zeros(n_win(1, h),32); % Average
    gamma_std{1, h} = zeros(n_win(1, h),32); % Standard Deviation
    gamma_mav{1, h} = zeros(n_win(1, h),32); % Mean Absolute Value
    gamma_rms{1, h} = zeros(n_win(1, h),32); % Root Mean Square
    gamma_skw{1, h} = zeros(n_win(1, h),32); % Skewness
    gamma_kur{1, h} = zeros(n_win(1, h),32); % Kurtosis
    gamma_hac{1, h} = zeros(n_win(1, h),32); % Hjorth Activity
    gamma_hmb{1, h} = zeros(n_win(1, h),32); % Hjorth Mobility
    gamma_hcp{1, h} = zeros(n_win(1, h),32); % Hjorth Complexity
    gamma_she{1, h} = zeros(n_win(1, h),32); % Shannon's EntropyW
    gamma_spe{1, h} = zeros(n_win(1, h),32); % Spectral Entropy
    gamma_psd{1, h} = zeros(n_win(1, h),32); % Power Spectral Density
end 

for h = 1:1:9
    w_start = 1;                % Start at the first sample
    w_end = w_size;             % End at window length
    for n = 1:1:n_win(1, h)
        gamma_avg{1, h}(n, :) = mean(eeg_gamma{1, h}(w_start:w_end, :));                   % Average for window
        gamma_std{1, h}(n, :) = std(eeg_gamma{1, h}(w_start:w_end, :));                    % Standard Deviation for window
        gamma_mav{1, h}(n, :) = mean(abs(eeg_gamma{1, h}(w_start:w_end, :)));              % Mean Absolute Value for window
        gamma_rms{1, h}(n, :) = rms(eeg_gamma{1, h}(w_start:w_end, :));                    % Root Mean Square for window
        gamma_skw{1, h}(n, :) = skewness(eeg_gamma{1, h}(w_start:w_end, :));               % Skewness for window
        gamma_kur{1, h}(n, :) = kurtosis(eeg_gamma{1, h}(w_start:w_end, :));               % Kurtosis for window
        gamma_hac{1, h}(n, :) = var(eeg_gamma{1, h}(w_start:w_end, :));                    % Hjorth Activity for window
        [gamma_hmb{1, h}(n, :), gamma_hcp{1, h}(n, :)] = ...
            HjorthParameters(eeg_gamma{1, h}(w_start:w_end, :));                         % Hjorth Mobility and Complexity for window
        gamma_she{1, h}(n, :) = wentropy(eeg_gamma{1, h}(w_start:w_end, :), 'shannon');    % Shannon's Entropy for window
        gamma_spe{1, h}(n, :) = wentropy(eeg_spec_gamma{1, h}(w_start:w_end, :), 'shannon');     % Spectral Entropy for window
        gamma_psd{1, h}(n, :) = mean(pburg(eeg_gamma{1, h}(w_start:w_end, :), 4));               % Power Spectral Density for window
    
        w_start = w_start + overlap;
        w_end = w_start + w_size;
    end
end

feats = {gamma_avg, gamma_std, gamma_mav, gamma_rms, gamma_skw, gamma_kur, gamma_hac, ...
    gamma_hmb, gamma_hcp, gamma_she, gamma_spe, gamma_psd};

clearvars h n gamma_avg gamma_std gamma_mav gamma_rms gamma_skw gamma_kur gamma_hac ...
    gamma_hmb gamma_hcp gamma_she gamma_spe gamma_psd eeg_gamma eeg_spec_gamma ...
    overlap w_end w_start w_size

%% Export features for the gamma band

for h = 1:1:9
    eeg_features{1, h} = table('Size', [n_win(1, h)*12 ,34], ...
        'VariableTypes', var_types);                                                % Create table to store all features
    allVars = 1:width(eeg_features{1, h});
    eeg_features{1, h} = renamevars(eeg_features{1, h}, allVars, table_names);      % Column names
    eeg_features{1, h}(:, 1) = {p};

    for n = 1:1:12
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 2) = {labels(1, n)};  % Label features on table
        feat = array2table(real(feats{1, n}{1, h}));
        eeg_features{1, h}(n_win(1, h)*(n-1)+1:n_win(1, h)*n, 3:34) = feat;         % Label features on table
    end
    g = "P" + p + "\Features_P" + p + "_S" + h +"_Gamma.csv";                                  % Make a csv file output name
    writetable(eeg_features{1, h}, g);
end 

clearvars h allVars eeg_features feat feats n g n_win labels p table_names ...
    table_size var_types
