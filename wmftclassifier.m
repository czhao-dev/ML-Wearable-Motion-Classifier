function [motion] = wmftclassifier( file )

%List of WMFT movements
movements = ['WMFT 1: Forearm to table        ';
             'WMFT 2: Forearm to box          ';
             'WMFT 3: Extend elbow            ';
             'WMFT 4: Extend elbow with weight';
             'WMFT 5: Hand to table           ';
             'WMFT 6: Hand to box             ';
             'WMFT 7: Weight to box           ';
             'WMFT 8: Reach and retrieve      ';
             'WMFT 9: Lift can                ';
             'WMFT 10: Lift pencil            ';
             'WMFT 11: Lift paper clip        ';
             'WMFT 12: Stack checkers         ';
             'WMFT 13: Flip cards             ';
             'WMFT 14: Grip strength          ';
             'WMFT 15: Turn key in lock       ';
             'WMFT 16: Fold towel             ';
             'WMFT 17: Lift basket            ';
             'Undefined                       ';];
movementsdata = cellstr(movements);

data= ReadData(file);                       %aquire data
i= DetermineInitialWindow(data);            %find the initial window
[acc,acc_mean]= preprocess(data, i);                   %find acc
data = [data acc];
d = trajectory_estimation(acc, data);       %aquire trajectory estimation data


x_diff = d(end,1)-d(1,1);
y_diff = d(end,2)-d(1,2);
z_diff = d(end,3)-d(1,3);

t = 1:1:length(d(:,3));

x = d(:,1);
y = d(:,2);
z = d(:,3);
x_acc = acc(:,1);
y_acc = acc(:,2);
z_acc = acc(:,3);
[azimuth,elevation,r]=cart2sph(x,y,z);
[azimuth_acc,elevation_acc,r_acc]=cart2sph(x_acc,y_acc,z_acc);
x_power=average_power(x);
y_power=average_power(y);
azimuth_power=average_power(azimuth);
elevation_power=average_power(elevation);
r_power=average_power(r);
z_power=average_z_power(z);
z_peaks_min=findpeaks(-z);
z_peaks_max=findpeaks(z);
az_peaks=findpeaks(azimuth);
z_mean = mean(z);
r_acc_power = acc_power(r_acc);

x_power_threshold = 0.001;
y_power_threshold = 0.002;
z_power_threshold_upper = 0.3;
z_power_threshold_lower = 0.1;
azimuth_power_threshold_1lb = 65;
elevation_var_threshold = 0.3;
r_power_threshold_lower = 1.5;
r_power_threshold_upper = 6;

[azimuth_diff,elevation_diff,r_diff]=cart2sph(x_diff,y_diff,z_diff);
azimuth_var = var(azimuth);
elevation_var = var(elevation);

if z_power > z_power_threshold_upper
    if length(z_peaks_min)<6 || length(az_peaks)<5;
        if abs(min(z)-z(end)) > 0.02   % need to distinguish motion that goes up and then down
            if elevation_var > elevation_var_threshold % little turning motion
                if r_power > r_power_threshold_lower
                    motion=movementsdata{6};
                else motion=movementsdata{5};
                end
            else
                if r_power > r_power_threshold_upper
                    motion=movementsdata{17};
                else
                    if (azimuth_diff/elevation_diff)<0.5
                        motion=movementsdata{7};
                    elseif  (azimuth_diff/elevation_diff)<0.66
                        motion=movementsdata{2};
                    else motion=movementsdata{1};
                    end
                end
            end
        else
            if z_power > 9
                motion=movementsdata{9};
            elseif z_power > 3 && z_power < 5
                motion=movementsdata{10};
            else motion=movementsdata{11};
            end
        end
    else
        if azimuth_var > 0.3 && azimuth_var < 0.7
            motion=movementsdata{13};
        elseif azimuth_var > 0.7 && azimuth_var < 1
            motion=movementsdata{16};
        else motion=movementsdata{12};
        end
    end
elseif z_power < z_power_threshold_lower
    if x_power < x_power_threshold && y_power < y_power_threshold
        motion=movementsdata{14};
    else
        if azimuth_diff > 0
            motion=movementsdata{8};
        else
            if azimuth_power > azimuth_power_threshold_1lb
                if length(z_peaks_min)>=3
                    motion=movementsdata{15};
                else motion=movementsdata{4};
                end
            else motion=movementsdata{3};
            end
        end
    end
else motion=movementsdata{18};
end

end

% read data from file
function data = ReadData(file)
as = 2048;       % sensitivity of 16g accelerometer
gs = 16.4;       % sensitivity of gyroscope
qs = 1073741824; % sensitivity of quaternion
fid = fopen(file,'r');
raw_data = [];
id = 0;
a = fgets(fid);
while(ischar(a))
    id = id + 1;
    if id == 1
        a = fgets(fid);
        continue
    else
        raw_data = [raw_data; str2num(a)];
    end
    a = fgets(fid);
end
fclose(fid);
if ~isempty(raw_data)
    raw_data(:,2) = -raw_data(:,2);
    raw_data(:,3) = -raw_data(:,3);
    acc = raw_data(:,2:4)/as;
    gyro = raw_data(:,5:7)/gs;
    q = raw_data(:,8:11)/qs;
    data = [acc gyro q];
else
    data = [];
end
end

% determine the window for initialization
function i = DetermineInitialWindow(data)
wlen = 20;
int_threshold = 0.003;
wlen = 2*wlen;
for i = 1:wlen+1:length(data(:,1))
    acc = data(i:i+wlen,:);
    acc_var = var(acc(:,1).^2 + acc(:,2).^2 + acc(:,3).^2);
    if (acc_var >= int_threshold)
        break;
    end
end
i = i - 1;
if i == 0
    i = 1;
else
    i = i - wlen/2;
end
acc = data(:,1:3);
gyro = data(:,4:6);
q = data(:,7:10);
end

% map acceleration to global coordinate and perform gravity subtraction
function [acc,acc_mean] = preprocess(data, wlen)
acc = data(1:wlen,1:3);
q = data(1:wlen,7:10);
global a_ref;
a_ref = mean(acc,1);
acc_mean = a_ref;
q_ref = mean(q,1);
acc = data(:,1:3);
q = data(:,7:10);
q = quatdivide(q, q_ref);
[acc] = gravity_subtraction(acc, a_ref, q);
end

% subtrac gravity
function [global_acc] = gravity_subtraction(acc, acc_mean, q)
global_acc = zeros(size(acc));
for i = 1:length(acc)
    a_temp = [0 acc(i,:)];
    a_temp = quatmultiply(quatmultiply(q(i,:),a_temp),quatconj(q(i,:)));
    global_acc(i,:) = a_temp(2:4) - acc_mean;
end
end

% estimate foot motion
function d = trajectory_estimation(acc, data)
dt = 1/200;
a = acc*9.8;
v = cumsum(a)*dt;
v1 = zero_velocity_update(data, v);
d = cumsum(v1)*dt;

%{
d_sph = zeros(size(d_cart));
for i = 1:length(d_sph(:,1))
    
    x=d_cart(i,1);
    y=d_cart(i,2);
    z=d_cart(i,3);
    
    r = sqrt(x^2 + y^2 + z^2);
    azimuth = atan(y/x);
    elevation = acos(z/r);
    
    d_sph(i,1)=azimuth;
    d_sph(i,2)=elevation;
    d_sph(i,3)=r;
    
end
%}

global a_ref;
global firstMotionSeg;
[reConstructedD] = FrameConstruction(a_ref, d, firstMotionSeg); %a_ref, zv_window(1,2) zv_window(2,1)

end

% code for zero velocity update
function v1 = zero_velocity_update(data, v)
% data(:,1:3) accelerometer measurements
% data(:,4:6) gyroscope measurements
% data(:,7:10) quaternion representing orientations
% data(:,11:13) acceleration after gravity correction

acc = data(:,11:13);
gyro = data(:,4:6);
rawAcc = data(:,1:3);
zv_window = process_zv(acc,gyro,rawAcc); % detect zero velocity windows

global firstMotionSeg;
firstMotionSeg=[zv_window(1,2) zv_window(2,1)]; %DEBUG: first motion segment

v1 = zeros(size(v));
j = 1;
for i = 1:length(v(:,1))
    if i >=zv_window(j,1) && i<zv_window(end,2)
        if (i > zv_window(j+1,2))
            j=j+1;
        end
        % only softupdate v in the motion part, and force the v in
        % stationary part to be zero
        if(i>=zv_window(j,2) && i<=zv_window(j+1,1))
            vDrift=(v(zv_window(j+1,1),:)*(i-zv_window(j,2))+v(zv_window(j,2),:)*(zv_window(j+1,1)-i))/(zv_window(j+1,1)-zv_window(j,2));
            v1(i,:) = v(i,:) - vDrift;
        else
            v1(i,:)=0;
        end
    end
end
end

function zv_window = process_zv(acc, gyro, rawAcc)
wlen = 40;
% CHOOSE YOUR OWN THRESHOLDS OF motionAcc, gyro, AND rawAcc.
% set the threshold to be Inf if you don't want to use that measurement to detect zv window
threshold = [0.02 10 Inf];

%define and evaluate the energy (you can try different ways to define the energy)
%figure;
acc_energy = acc(:,1).^2 + acc(:,2).^2 + acc(:,3).^2;

%subplot(3,1,1);plot(acc_energy);
%title('acc_energy = acc(:,1).^2 + acc(:,2).^2 + acc(:,3).^2');

gyro_energy = abs(gyro(:,1)) + abs(gyro(:,2)) + abs(gyro(:,3));
%subplot(3,1,3); plot(gyro_energy);
%title('gyro_energy = abs(gyro(:,1)) + abs(gyro(:,2)) + abs(gyro(:,3))');

rawAcc_energy = abs(rawAcc(:,1).^2 + rawAcc(:,2).^2 + rawAcc(:,3).^2-1);
%subplot(3,1,2); plot(acc_energy);
%title('rawAcc_energy = abs(rawAcc(:,1).^2 + rawAcc(:,2).^2 + rawAcc(:,3).^2-1)');

% collect zero velocity point
zv_point = collect_zvpoint(acc_energy, gyro_energy, rawAcc_energy, wlen, threshold);

% merge zero velocity points into zero velocity window
zv_window = merge_zvpoint(zv_point);

zv_wlen = [];
for i = 2:length(zv_window(:,1))-1
    if zv_window(i,2) - zv_window(i,1) > 2*wlen
        %         zv_window(i,1) = round(mean([zv_window(i,1) zv_window(i,2)])) - wlen;
        %         zv_window(i,2) = round(mean([zv_window(i,1) zv_window(i,2)])) + wlen;
        zv_window(i,1) = zv_window(i,1) + wlen;
        zv_window(i,2) = zv_window(i,2) - wlen;
    end
    zv_wlen = [zv_wlen; zv_window(i,2) - zv_window(i,1)];
end
zv_wlen_mean = round(mean(zv_wlen));
if zv_window(1,2) - wlen > zv_window(1,1)
    zv_window(1,2) = zv_window(1,2) - wlen;
    if zv_window(1,2) - zv_wlen_mean > zv_window(1,1)
        zv_window(1,1) = zv_window(1,2) - zv_wlen_mean;
    end
end
if zv_window(end,1) + wlen < zv_window(end,2)
    zv_window(end,1) = zv_window(end,1) + wlen;
    if zv_window(end,1) + zv_wlen_mean < zv_window(end,2)
        zv_window(end,2) = zv_window(end,1) + zv_wlen_mean;
    end
end
end

function zv_point = collect_zvpoint(acc_energy, gyro_energy, rawAcc_energy, wlen, threshold)
zv_point = [];

acc_energy_mean = zeros(size(acc_energy));
gyro_energy_mean = zeros(size(gyro_energy));
rawAcc_energy_mean = zeros(size(rawAcc_energy));

for i = 1:length(acc_energy)
    window_start = i - wlen;
    window_end = i + wlen;
    if window_start < 1
        window_start = 1;
    end
    if window_end > length(acc_energy)
        window_end = length(acc_energy);
    end
    
    acc_energy_mean(i) = mean(acc_energy(window_start:window_end));
    gyro_energy_mean(i) = mean(gyro_energy(window_start:window_end));
    rawAcc_energy_mean(i) = mean(rawAcc_energy(window_start:window_end));
    if (acc_energy_mean(i) < threshold(1) && gyro_energy_mean(i)< threshold(2) && rawAcc_energy_mean(i) < threshold(3)) % set your own
        zv_point = [zv_point window_start:window_end];
    end
end
zv_point = unique(zv_point);
end

function zv_window = merge_zvpoint(zv_point)
zv_window = [];
i = 1;
window_start = zv_point(i);
window_point = window_start;
while i < length(zv_point)
    i = i + 1;
    if (zv_point(i) - window_point > 1)
        window_end = zv_point(i-1);
        zv_window = [zv_window; [window_start window_end]];
        window_start = zv_point(i);
        window_point = window_start;
    else
        window_point = zv_point(i);
    end
end
window_end = zv_point(i);
zv_window = [zv_window; [window_start window_end]];
end

%function [pos SL] = FrameConstruction(acc_mean, pos, center)
function [pos] = FrameConstruction(acc_mean, pos, center)
zaxis = acc_mean/norm(acc_mean);
xaxis = (pos(center(2),:) - pos(center(1),:))/norm(pos(center(1),:) - pos(center(2),:));
yaxis = cross(xaxis, zaxis);
yaxis = yaxis/norm(yaxis);
xaxis = cross(zaxis, yaxis);
xaxis = xaxis/norm(xaxis);
R = [xaxis; yaxis; zaxis]\eye(3);
pos = pos*R;

% SL = zeros(length(center)-1,1);
% for i = 1:length(center)-1
%     SL(i) = norm(pos(center(i+1),:) - pos(center(i),:));
% end
end

%CountIndex function counts the number of useful data points
function [countindex]=CountIndex(d,i)           
countindex=1;
erroroffset=.0005;
for start=i:length(d);
    countindex=countindex+1;
    if ((start>i+125)&&(abs(d(start,1)-d(start-5,1))<erroroffset)&&(abs(d(start,2)-d(start-5,2))<erroroffset)&&(abs(d(start,3)-d(start-5,3))<erroroffset)&&(abs(d(start-5,1)-d(start-10,1))<erroroffset)&&(abs(d(start-5,2)-d(start-10,2))<erroroffset)&&(abs(d(start-5,3)-d(start-10,3)))<erroroffset)
        break;
    end
end

end

function [p] = average_z_power(data)
e=0;

for i=2:length(data(:,1))
    if data(i-1,1)~=data(i,1)
        if data(i,1) < 0
            e = e+(abs(data(i,1)-data(1,1)))^2;
        end
    end
end

p = (e/length(data(:,1)))*100;
end

function [p] = average_power(data)
e=0;

for i=2:length(data(:,1))
    if data(i-1,1)~=data(i,1)
       e = e+(abs(data(i,1)-data(1,1)))^2;
    end
end

p = (e/length(data(:,1)))*100;
end

function [p] = acc_power(data)
e=0;

for i=2:length(data(:,1))
    if data(i-1,1)~=data(i,1)
       e = e+(abs(data(i,1)))^2;
    end
end

p = (e/length(data(:,1)))*100;
end
