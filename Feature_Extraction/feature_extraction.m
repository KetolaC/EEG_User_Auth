clear all;
eeg_sample = readmatrix('HS_P7_S1_EEG.csv');
Sample_rate = 500; %sampling frequency of the original data
dt = 1/Sample_rate; %time per sample 
N = 307632; %number of samples per channel
Ts = 0:N-1; %interval created or the time axis
Ts = Ts.*dt;
f0 = Sample_rate/N; %frequency per sample


%% Viewing the raw EEG data 
 figure
 plot(eeg_sample) 
 title('EEG data raw')
 xlabel('Samples'), ylabel('Magnitude'), grid on 

 %% Viewing the time scale versus the raw EEG data 
 figure
 plot(Ts, eeg_sample) 
 title('Time versus EEG data raw')
 xlabel('Time(s)'), ylabel('Magnitude'), grid on 


 EEGf = fft(eeg_sample); %transforming to frequency domain
 
 %% Viewing the frequency domain data
 figure
 plot(f0*(0:N-1),abs(EEGf)) %f0 is 0.2 per sample and it is same for each. You are plotting freq for all samples
 title('Frequency spectrum of the Raw EEG signal')
 xlabel('frequency(Hz)'), ylabel('Energy'), grid on
 %xlim([0 125]);


cb=0.2;
ca=50;
[b,a] = butter(4,[cb*2/Sample_rate ca*2/Sample_rate]);
EEG_BW = filter(b,a,eeg_sample);%applying butter worth filter

%% Plotting the frequency spectrum of the filtered data
EEG_BWf=fft(EEG_BW);
figure
plot(f0*(0:N-1),abs(EEG_BWf))
title('Frequency spectrum of the filtered EEG signal (0.1Hz a 100Hz)')
xlabel('Frequency(Hz)'),ylabel('Energy'),grid on
xlim([0 80]);
EEG_BWf = real(EEG_BWf);




%% WINDOWING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

L=250;
SV=round(L/2); %50 perc overlap
EEG = EEG_BW;
%Tms=time*1000;
Tms = N*dt*1000;
W=floor(Tms/(L-SV));

%% Seperate the Frequency Bands



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

