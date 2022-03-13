clear;
clc;

for p = 4:12        %change participants here
    band2comb = {'Alpha'; 'Beta'; 'Delta'; 'Gamma'; 'Theta'; 'EEG Data'};
    for h = 1:5
        combTable = concatSessionData(9, p, char(band2comb(h)));
        filename = ['Features_P', num2str(p), '_', char(band2comb(h)), '.csv'];
        writetable(combTable, filename);
    end

    combTable = concatSessionData(9, p, char(band2comb(6)));
    filename = ['Features_P', num2str(p), '.csv'];
    writetable(combTable, filename);
end
