clear;
clc;

band2sec = {'Alpha'; 'Beta'; 'Delta'; 'Gamma'; 'Theta'};
for p = 4:12        %change participants here
    for h = 1:5
        [trainDat, validDat, testDat] = sectionData(p, char(band2sec(h)));
        trainDatFilename = ['Features_P', num2str(p), '_', char(band2sec(h)), '_TrainingData.csv'];
        writetable(trainDat,trainDatFilename);
        validDatFilename = ['Features_P', num2str(p), '_', char(band2sec(h)), '_ValidationData.csv'];
        writetable(validDat,validDatFilename);
        testDatFilename = ['Features_P', num2str(p), '_', char(band2sec(h)), '_TestingData.csv'];
        writetable(testDat,testDatFilename);
    end

    [trainDat, validDat, testDat] = sectionData(p, char(band2sec(h)));
    trainDatFilename = ['Features_P', num2str(p), '_TrainingData.csv'];
    writetable(trainDat,trainDatFilename);
    validDatFilename = ['Features_P', num2str(p), '_ValidationData.csv'];
    writetable(validDat,validDatFilename);
    testDatFilename = ['Features_P', num2str(p), '_TestingData.csv'];
    writetable(testDat,testDatFilename);
    
    disp(['Participant ', num2str(p), ' completed'])
end
