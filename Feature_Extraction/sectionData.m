function [trainDat, validDat, testDat] = sectionData(part, band)
%This is a function that takes the combined session data from a participant
%and separates it into 3 parts
%Inputs:
%   part: the participant number (used for reading the data files)
%   band: the band (used for reading the data files
%Outputs:
%   trainDat: the training data (80% of original)
%   validDat: the validation data (10% of original)
%   testDat: the testing data (10% of the original)

%Load table
% namedBands = {'Alpha'; 'Beta'; 'Delta'; 'Gamma'; 'Theta'};
% findBand = 0;
% for h = 1:1:5
%     if strcmp(band, namedBands(h))
%         findBand = findBand + 1;
%     end
% end
% if findBand == 1
%     filename = ['Features_P', num2str(part), '_', band, '.csv'];
% else
%     filename = ['Features_P', num2str(part), '.csv'];
% end
if strcmp(band,'')
    filename = ['Features_P', num2str(part), '.csv'];
else
    filename = ['Features_P', num2str(part), '_', band, '.csv'];
end
table2Sec = readtable(filename);

[datRow,~] = size(table2Sec);
randInds = randperm(datRow,datRow).';
ranDat = table2Sec(randInds(1),:);
ranDat([1],:) = table2Sec([randInds(1)],:);
for h = 2:datRow
    ranDat([h],:) = table2Sec([randInds(h)],:);
end

dat10Per = round(datRow*0.1);
trainUpInd = datRow - 2*dat10Per;
trainDat = ranDat([1:trainUpInd],:);
validUpInd = trainUpInd+dat10Per;
validDat = ranDat([trainUpInd+1:validUpInd],:);
testDat = ranDat([validUpInd+1:datRow],:);

end
