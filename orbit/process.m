file_name = '20SA_15DAY_endtime_550km.csv';
t = readtable(file_name);
sz = 1;
while sz <= size(t,1)
    tmp = cell2mat(t.Source(sz));
    tmp = strsplit(tmp, '_');
    tmp = str2num(cell2mat(tmp(2)));
    if tmp > 10
        t(sz,:) = [];
    else
        sz = sz + 1;
    end
end
write_file_name = '20SA_15DAY_endtime_550km_adjust.csv';
writetable(t, write_file_name);
