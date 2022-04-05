function [trainDat, validDat, testDat] = sectionData(part, band)
% This is a function that takes the combined session data from a participant
% and separates it into 3 parts
%
% Inputs:
%   part: the participant number (used for reading the data files)
%   band: the band (used for reading the data files
% Outputs:
%   trainDat: the training data (80% of original)
%   validDat: the validation data (10% of original)
%   testDat: the testing data (10% of the original)

% part = 2;
% band = '';

    trainTable = [];    % Cell for the complete training data
    validTable = [];    % Cell for the complete validation data
    testTable = [];     % Cell for the complete testing data
    rng(45);            % Seed the random number generator for replicability
    
    for sec = 1:9       % Repeat for all sections
    
        if strcmp(band,'')
            filename = "P" + part + "\Features_P" + num2str(part) + "_S" + sec + ".csv";                % Filename is no band is given
        else 
            filename = "P" + part + "\Features_P"  + num2str(part) + "_S" + sec + "_" + band + ".csv";  % Filename when a band is given
        end
        
        table2Sec = readtable(filename);        % Read the .csv file
        cell2sec = table2cell(table2Sec);
        [datRow,~] = size(cell2sec);           % Get the rows of the table
    
        randInds = randperm(datRow,datRow);     % Create a vector for randomizing the table rows
        ranDat = cell(datRow, 34);
    
        for h = 1:datRow
            ranDat(h, :) =  cell2sec(randInds(h), :); % Create table with randomly shuffled rows   
        end

%         trainTable = vertcat(trainTable, ranDat(1:round(datRow*0.8), :));       % Assign 80% of data to training
%         validTable = vertcat(validTable, ranDat(round(datRow*0.8)+1:round(datRow*0.9), :)); %  Assign 10% of data to validation
%         testTable = vertcat(testTable, ranDat(round(datRow*0.9)+1:end, :));     % Assign 10% of data to testing

        trainTable = [trainTable; ranDat(1:round(datRow*0.8), :)];       % Assign 80% of data to training
        validTable = [validTable; ranDat(round(datRow*0.8)+1:round(datRow*0.9), :)]; %  Assign 10% of data to validation
        testTable = [testTable; ranDat(round(datRow*0.9)+1:end, :)];     % Assign 10% of data to testing
    
    end
    
    table_names = ["Participant", "Feature", "ch1", "ch2", "ch3", "ch4", ...
            "ch5", "ch6",  "ch7", "ch8", "ch9", "ch10", "ch11", "ch12", ...
            "ch13", "ch14", "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", ...
            "ch21", "ch22", "ch23", "ch24", "ch25", "ch26", "ch27", "ch28", ...
            "ch29", "ch30", "ch31", "ch32"]; 

     trainDat = cell2table(trainTable, "VariableNames", table_names);
     validDat = cell2table(validTable, "VariableNames", table_names);
     testDat = cell2table(testTable, "VariableNames", table_names);

end
