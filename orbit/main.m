day_list = [15];
for day = day_list
    startTime = datetime(2023, 6, 1, 00, 00, 0);
    stopTime = startTime + days(day);
    sampleTime = 60; % seconds
    sc = satelliteScenario(startTime, stopTime, sampleTime);
    
    % shanghai 经纬度
    lat = 31.2304;
    lon = 121.4737;
    gs = groundStation(sc, lat, lon);
    
    height = 1100e3; % m
    R_earth = 6.37814e+06; % m
    radius = height + R_earth; % m
    inc = 53;
    total_sat = 20;
    total_plan = 5;
    F = 1;
    RAAN = 10;
    sat = walkerDelta(sc, radius, inc, total_sat, total_plan, F, RAAN=RAAN);
    
    ac = access(sat, gs);
    intvls = accessIntervals(ac);
    intvls = sortrows(intvls, 5);
    file_name = [num2str(total_sat), 'SA_' , num2str(day), 'DAY', '_endtime', '_1100km', '.csv'];
    writetable(intvls, file_name);
end
