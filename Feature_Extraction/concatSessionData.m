function concatTable = concatSessionData(numSess, part, band)
%This is a function that concatenates all session data from a single user
%Inputs:
%   numSess: the number of sessions that need to be concatenated
%   part: the participant number (used for reading and writing csv files)
%   band: the band (used for reading and writing csv files)
%
%Output:
%   newTable: the new table with the concatenated data

namedBands = {'Alpha'; 'Beta'; 'Delta'; 'Gamma'; 'Theta'};
findBand = 0;
for h = 1:1:5
    if strcmp(band, namedBands(h))
        findBand = findBand + 1;
    end
end

sessDatLen = zeros(1,numSess);
if findBand == 1
    filename = ['Features_P', num2str(part), '_S', num2str(1), '_', band, '.csv'];
else
    filename = ['Features_P', num2str(part), '_S', num2str(1), '.csv'];
end
concatTable = readtable(filename);
[rowSess,~] = size(readtable(filename));
sessDatLen(1) = rowSess;

for h = 2:1:numSess
    if findBand == 1
        filename = ['Features_P', num2str(part), '_S', num2str(h), '_', band, '.csv'];
    else
        filename = ['Features_P', num2str(part), '_S', num2str(h), '.csv'];
    end
    concatTable = vertcat(concatTable,readtable(filename));
    [rowSess,~] = size(readtable(filename));
    sessDatLen(h) = rowSess;
end

[col,~] = size(concatTable);
Session = zeros(col,1);
startInd = 1;
for h = 1:1:numSess
    Session(startInd:startInd+sessDatLen(h)-1) = h;
    startInd = startInd + sessDatLen(h);
end
sessNumTable = table(Session);

concatTable = [concatTable sessNumTable];
concatTable = movevars(concatTable,'Session','After','Participant');
end
