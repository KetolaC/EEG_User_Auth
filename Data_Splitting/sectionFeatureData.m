clear;
clc;

band2sec = {''; 'Alpha'; 'Beta'; 'Delta'; 'Gamma'; 'Theta'};
trainBandTables = {[]; []; []; []; []; []};
validBandTables = {[]; []; []; []; []; []};
testBandTables = {[]; []; []; []; []; []};

for p = 1:12        % Go through all 12 participants
    disp("Participant " + p)
    for h = 1:6     % Go through combined and individual band data
        disp("Band: " + h)
        [trainDat, validDat, testDat] = sectionData(p, band2sec(h));
%         trainBandTables{h} = vertcat(trainBandTables{h}, trainDat);
%         validBandTables{h} = vertcat(validBandTables{h}, validDat);
%         testBandTables{h} = vertcat(testBandTables{h}, testDat);
        
        trainBandTables{h} = [trainBandTables{h}; trainDat];
        validBandTables{h} = [validBandTables{h}; validDat];
        testBandTables{h} = [testBandTables{h}; testDat];
    end
    
end

trainDatFilename = "TrainingData.csv";
writetable(trainBandTables{1},trainDatFilename);
validDatFilename = "ValidationData.csv";
writetable(validBandTables{1},validDatFilename);
testDatFilename = "TestingData.csv";
writetable(testBandTables{1},testDatFilename);

for h = 2:6
    trainDatFilename =  band2sec(h) + "_TrainingData.csv";
    writetable(trainBandTables{h},trainDatFilename);
    validDatFilename = band2sec(h) + "_ValidationData.csv";
    writetable(validBandTables{h},validDatFilename);
    testDatFilename = band2sec(h) + "_TestingData.csv";
    writetable(testBandTables{h},testDatFilename);
end

disp("Data Splitting Completed")

