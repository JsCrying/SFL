file_name_1 = '20SA_15DAY_endtime_550km_adjust.csv';
file_name_2 = '20SA_15DAY_endtime_1100km_adjust.csv';
t1 = readtable(file_name_1);
t2 = readtable(file_name_2);

union_table = union(t1, t2, 'sorted');
union_table = sortrows(union_table, 5);
write_file_name = '20SA_15DAY_endtime_550+1100km.csv';
writetable(union_table, write_file_name);