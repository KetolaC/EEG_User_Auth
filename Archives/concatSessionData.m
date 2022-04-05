function concatTable = concatSessionData(numSess, part, band)
%This is a function that concatenates all session data from a single user
%Inputs:
%   numSess: the number of sessions that need to be concatenated
%   part: the participant number (used for reading and writing csv files)
%   band: the band (used for reading and writing csv files)
%
%Output:
%   newTable: the new table with the concatenated data

%test values
numSess = 9;
part = 1;
band = 'Theta';

sessDatLen = zeros(1,numSess);
filename = ['Features_P', num2str(part), '_S', num2str(1), '_', num2str(band), '.csv'];
concatTable = readtable(filename);
[rowSess,~] = size(readtable(filename));
sessDatLen(1) = rowSess;

for h = 2:1:numSess
    filename = ['Features_P', num2str(part), '_S', num2str(h), '_', num2str(band), '.csv'];
    concatTable = vertcat(concatTable,readtable(filename));
    [rowSess,~] = size(readtable(filename));
    sessDatLen(h) = rowSess;
end

[col,~] = size(concatTable);
Session = zeros(col,1);
startInd = 1;
for h = 1:1:numSess
    for l = 1:1:sessDatLen(h)
        Session(startInd:startInd+sessDatLen(h)) = h;
    end
    startInd = sessDatLen(h);
end
sessNumTable = table(Session);

concatTable = [concatTable sessNumTable];
concatTable = movevars(concatTable,'Session','After','Participant');
end
