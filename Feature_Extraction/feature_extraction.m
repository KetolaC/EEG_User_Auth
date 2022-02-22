%% Load in ll the raw EEG data and variables

Sample_rate = 500; %sampling frequency of the original data
dt = 1/Sample_rate; %time per sample 

for n = 1:1:12
    for h = 1:1:9 %Get session 1 through 9 for participant
        g = "HS_P" + n + "_S" + h +".csv" %Make a csv file output name
        eeg_data{n, h} = readmatrix(g); % Create a row vector for each participant
        N(n, h) = length(eeg_data{n, h}); %number of samples per channel
        f0(n, h) = Sample_rate/N(n, h); %frequency per sample
    end 
end 

clearvars n h g 
 
%% Create a Butterworth filter for the overall EEG data

cb = 0.2; % High-pass frequency
ca = 50;  % Low-pass frequency
[b,a] = butter(4,[cb*2/Sample_rate, ca*2/Sample_rate]);  % Create a Butterworth filter

clearvars cb ca 

%% Filter the EEG data 

for n = 1:1:12
    for h = 1:1:9
        EEG_BW{n, h} = filter(b,a,eeg_data{n, h});  % Application of Butterworth filter
        EEG_BWf{n, h} = fft(EEG_BW{n, h}); % Convert to frequency domain 
    end 
end

clearvars a b  n h
%% Plot an instance of filtered EEG data
figure
plot(f0(1, 1)*(0:N(1, 1)-1),abs(EEG_BWf{1, 1}))
title('Frequency spectrum of the filtered EEG signal (0.1Hz a 100Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy'),grid on
xlim([0 80]);

%% Create filters for the five frequency bands 

hp = 0.2; % High-pass frequency
lp = 4;  % Low-pass frequency
[db,da] = butter(4,[hp*2/Sample_rate, lp*2/Sample_rate]);  % Delta band filter

hp = 4; % High-pass frequency
lp = 8;  % Low-pass frequency
[tb,ta] = butter(4,[hp*2/Sample_rate, lp*2/Sample_rate]);  % Theta band filter

hp = 8; % High-pass frequency
lp = 12;  % Low-pass frequency
[ab,aa] = butter(4,[hp*2/Sample_rate, lp*2/Sample_rate]);  % Alpha band filter

hp = 12; % High-pass frequency
lp = 26;  % Low-pass frequency
[bb,ba] = butter(4,[hp*2/Sample_rate, lp*2/Sample_rate]);  % Beta band filter

hp = 12; % High-pass frequency
lp = 26;  % Low-pass frequency
[gb,ga] = butter(4,[hp*2/Sample_rate, lp*2/Sample_rate]);  % Gamma band filter

clearvars lp hp
%% Apply frequency band filter on EEG data
for n = 1:1:12
    for h = 1:1:9
        EEG_delta{n, h} = filter(db,da,eeg_data{n, h});  % Extract delta band
        EEG_theta{n, h} = filter(tb,ta,eeg_data{n, h});  % Extract thata band
        EEG_alpha{n, h} = filter(ab,aa,eeg_data{n, h});  % Extract alpha band
        EEG_beta{n, h} = filter(bb,ba,eeg_data{n, h});  % Extract beta band
        EEG_gamma{n, h} = filter(gb,ga,eeg_data{n, h});  % Extract gamma band
        %EEG_BWf{n, h} = fft(EEG_BW{n, h}); % Convert to frequency domain 
    end 
end

clearvars n h db da ta tb aa ab bb ba ga gb

%% Extract real components
EEG_BWf{1, 1} = real(EEG_BWf{1, 1}); % Extract the real components of the EEG data

%% WINDOWING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L=250;
SV=round(L/2); %50 perc overlap
EEG = EEG_BW;
%Tms=time*1000;
Tms = N*dt*1000;
W=floor(Tms/(L-SV));


%% Feature Extraction

EEG_AVG= zeros(W,32); % Average
EEG_STD= zeros(W,32); % Standard Deviation
EEG_MAV= zeros(W,32); % Mean Absolute Value
EEG_RMS= zeros(W,32); % Root Mean Square
EEG_SKW= zeros(W,32); % Skewness
EEG_KUR= zeros(W,32); % Kurtosis
EEG_HAC= zeros(W,32); % Hjorth Activity
EEG_HMB= zeros(W,32); % Hjorth Mobility
EEG_HCP= zeros(W,32); % Hjorth Complexity
EEG_SHE= zeros(W,32); % Shannon's Entropy
EEG_PSD= zeros(W,32); % Power Spectral Density
EEG_SPE= zeros(W,32); % Spectral Entropy
EEG_AUR= zeros(W,32); % Autoregression
EEG_COR= zeros(W,32); % Coherence 
EEG_CCR= zeros(W,32); % Cross-correlation 
EEG_ABP= zeros(W,32); % Average Band Power

Start=1;
End=L;

for i = 1:W
    if End > N
        break;
    end
    EEG_AVG(i,:) = mean(EEG(Start:End,:));
    EEG_STD(i,:) = std(EEG(Start:End,:));
    EEG_MAV(i,:) = mean(abs(EEG(Start:End,:)));
    EEG_RMS(i,:) = rms(EEG(Start:End,:));
    EEG_SKW(i,:) = skewness(EEG(Start:End,:));
    EEG_KUR(i,:) = kurtosis(EEG(Start:End,:));
    [EEG_HMB(i,:), EEG_HCP(i,:)] = HjorthParameters(EEG(Start:End, :));
    %EEG_HAC = ;


    Start=Start+SV;
    End=End+SV;
end

%% NORMALIZE %%%%%%%%
EEG_AVG=EEG_AVG./max(EEG_AVG);
EEG_STD=EEG_STD./max(EEG_STD);
EEG_MAV=EEG_MAV./max(EEG_MAV);
EEG_RMS=EEG_RMS./max(EEG_RMS);
EEG_SKW=EEG_SKW./max(EEG_SKW);
EEG_KUR=EEG_KUR./max(EEG_KUR);

FEATS = [ EEG_AVG EEG_STD EEG_MAV EEG_RMS EEG_SKW EEG_KUR];

 figure
 plot(EEG_AVG(:,3)) 
 title('Features series 1 user 1')
 xlabel('Average'), ylabel('Values'), grid on 

 %% Append to csv 

