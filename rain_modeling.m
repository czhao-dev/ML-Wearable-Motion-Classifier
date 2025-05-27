addpath('C:\Users\Charles\Google Drive\Graduate School\MS\Spring 2017\EE 239AS\Project\slra-0.5\');

ncid = netcdf.open('C:\Users\Charles\Google Drive\Graduate School\MS\Spring 2017\EE 239AS\Project\pnwrain.50km.daily.4994.nc');
latitude = netcdf.getVar(ncid, 0);
longitude = netcdf.getVar(ncid, 1);
time = netcdf.getVar(ncid, 2);
data = netcdf.getVar(ncid, 3);

netcdf.close(ncid);

% Single value decomposition