latitude = netcdf.getVar(ncid, 0);
longitude = netcdf.getVar(ncid, 1);
time = netcdf.getVar(ncid, 2);
data = netcdf.getVar(ncid, 3);

netcdf.close(ncid);

% Single value decomposition
