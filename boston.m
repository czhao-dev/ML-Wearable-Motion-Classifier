% add path for the SLRA packet
addpath(fullfile(cd, 'slra-slra-c3aa24c'));
warning('off', 'all');
[PRCP_sys, SNOW_sys, TOBS_sys] = main();

% read weather data from the file
function [training_data, testing_data] = read_data()
    data_Reading = csvread('C:\Users\Charles\Desktop\962598.csv', 70136, 2, [70136, 2, 72082, 13]);

    % remove unnecessary columns
    % columns: DATE, PRCP, SNWD, SNOW, TMAX, TMIN, TOBS
    cols2remove = [2 3 7 8 9];
    data_Reading(:, cols2remove) = [];

    % remove 2/29/2012 from the data set
    data_Reading(60, :) = [];
    
    % remove invalid values in the data
    data_Reading = preprocess_data(data_Reading);

    % plot raw data
    % plot_data(data_Reading);
    
    % filter noise
    data_Reading = filter_noise(data_Reading);
    
    % plot filtered data
    % plot_data(data_Reading);
    
    % split data into training set (2013-2015) and testing set (2016)
    [training_data, testing_data] = split_data(data_Reading);
end

function [] = plot_data(data)
    % plot PRCP data
    figure
    plot(data(366:end, 2));
    ylabel('Precipitation (mm)');
    
    % plot SNOW data
    figure
    plot(data(366:end, 4));
    ylabel('Snow Fall (mm)');
    
    % plot TOBS data
    figure
    plot(data(366:end, 7));
    ylabel(['Temperature (', char(176), 'F)']);
end

function [data] = preprocess_data(data)
	for i = 1:size(data, 1)
		% Process invalid PRCP, SNWD, SNOW values
		if data(i, 2) == -9999
			data(i, 2) = 0;
		end
		if data(i, 3) == -9999
			data(i, 3) = 0;
		end
		if data(i, 4) == -9999
			data(i, 4) = 0;
		end

		% Process invalid TMAX values
		if (data(i, 5) == -9999)
			if ((data(i, 6) ~= -9999) && (data(i, 7) ~= -9999))
				data(i, 5) = 2*data(i, 7) - data(i, 6);
			elseif (i < 365)
				data(i, 5) = data(i+365, 5);
			else
				data(i, 5) = data(i-365, 5);
			end
		end

		% Process invalid TMIN values
		if (data(i, 6) == -9999)
			if ((data(i, 5) ~= -9999) && (data(i, 7) ~= -9999))
				data(i, 6) = data(i, 5) - 2*data(i, 7);
			elseif (i < 365)
				data(i, 6) = data(i+365, 6);
			else
				data(i, 6) = data(i-365, 6);
			end
		end

		% Process invalid TOBS values
		if (data(i, 7) == -9999)
			if ((data(i, 5) ~= -9999) && (data(i, 6) ~= -9999))
				data(i, 7) = (data(i, 5)+data(i, 6))/2;
			elseif (i < 365)
				data(i, 7) = data(i+365, 7);
			else
				data(i, 7) = data(i-365, 7);
			end
		end
	end
end

function [data] = filter_noise(data)
    coeff30dMA = ones(1, 30)/30;
    coeff65dMA = ones(1, 65)/65;
    coeff90dMA = ones(1, 90)/90;
    avg90dPRCP = filter(coeff90dMA, 1, data(:, 2));
    avg90d65dPRCP = filter(coeff65dMA, 1, avg90dPRCP);
    avg30dSNWD = filter(coeff30dMA, 1, data(:, 3));
    avg90dSNOW = filter(coeff90dMA, 1, data(:, 4));
    avg90d30dSNOW = filter(coeff30dMA, 1, avg90dSNOW);
    avg30dTMAX = filter(coeff30dMA, 1, data(:, 5));
    avg30dTMIN = filter(coeff30dMA, 1, data(:, 6));
    avg30dTOBS = filter(coeff30dMA, 1, data(:, 7));
    data = [data(:, 1), 20*avg90d65dPRCP, avg30dSNWD, 20*avg90d30dSNOW, avg30dTMAX, avg30dTMIN, avg30dTOBS];
end

function [training_data, testing_data] = split_data(data)
	training_data = data(366:4*365, :);
	testing_data = data((4*365+1):(5*365), :);
end

function [M_arr] = ident_svd(training_data, testing_data)
    % calculate SVD on PRCP, SNOW, TOBS data
	[U_PRCP, D_PRCP, V_PRCP] = calculate_svd(training_data(:, 2));
    [U_SNOW, D_SNOW, V_SNOW] = calculate_svd(training_data(:, 4));
    [U_TOBS, D_TOBS, V_TOBS] = calculate_svd(training_data(:, 7));
    
    % plot PRCP, SNOW, TOBS SVD values with log y-axis
    % plot_svd(D_PRCP, D_SNOW, D_TOBS);
    
    % truncate D and V hankel
    [hankel_PRCP] = truncate(U_PRCP, D_PRCP, V_PRCP);
    [hankel_SNOW] = truncate(U_SNOW, D_SNOW, V_SNOW);
    [hankel_TOBS] = truncate(U_TOBS, D_TOBS, V_TOBS);
    
    lmax = 4;
    t = 365;
    days = 1:1:(365+325+lmax-1);
    ud = sin(((days-79.5)/365)*(2*pi));
    H_PRCP = calculate_H(ud, hankel_PRCP, lmax, t);
    H_SNOW = calculate_H(ud, hankel_SNOW, lmax, t);
    H_TOBS = calculate_H(ud, hankel_TOBS, lmax, t);
    
    % realize the system models
    sys_PRCP = h2ss(H_PRCP, 4);
    sys_SNOW = h2ss(H_SNOW, 4);
    sys_TOBS = h2ss(H_TOBS, 4);
    
    % plot comparison of model with testing data
    H = [H_PRCP, H_SNOW, H_TOBS];
    sys = [sys_PRCP, sys_SNOW, sys_TOBS];
    M_arr = plot_svd_sys_model(H, sys, training_data, testing_data);
end

function [H_truncated] = truncate(U, D, V)
    D_truncated = D(:, 1:325);
    V_truncated = V(1:325, 1:325);
    H_truncated = U*D_truncated*(V_truncated');
end

function [M_arr] = plot_svd_sys_model(H, sys, training_data, testing_data)
    M_arr = zeros(1, 3);
    days = 1:1:2*365;
    ud = sin(((days-79.5)/365)*(2*pi));
    
    % plot PRCP model
    H_PRCP = H(:, 1);
    PRCP_svd = 18*conv(H_PRCP(1:90), ud) + 1.75;
    figure
    plot(testing_data(:, 2), 'color', [0.65 0.65 0.65]);
    hold on
    plot(PRCP_svd(366-40:2*365-40), 'color', [0 0.3 1]);
    xticks([1, 25:25:325, 365]);
    ylabel('Precipitation (mm)');
    legend('PRCP data (2016)', 'Model');
    M_arr(1) = normalized_misfit(testing_data(:, 2), PRCP_svd(366-40:2*365-40));

    % plot SNOW model
    H_SNOW = H(:, 2);
    SNOW_svd = 1.75*conv(H_SNOW(1:180), ud);
    for i = 1:length(SNOW_svd)
        if SNOW_svd(i) < 0
            SNOW_svd(i) = 0;
        end
    end
    figure
    plot(testing_data(:, 4), 'color', [0.65 0.65 0.65]);
    hold on
    plot(SNOW_svd(30:365), 'color', [0 0.3 1]);
    xticks([1, 25:25:325, 365]);
    ylabel('Snow Fall (mm)');
    legend('SNOW data (2016)', 'Model');
    M_arr(2) = normalized_misfit(testing_data(:, 4), SNOW_svd(30:365+30));
    
    % plot TOBS model
    H_TOBS = H(:, 3);
    TOBS_svd = 15*conv(H_TOBS(1:90), ud) + mean(training_data(:, 7));
    figure
    plot(testing_data(:, 7), 'color', [0.65 0.65 0.65]);
    hold on
    plot(TOBS_svd(1:365), 'color', [0 0.3 1]);
    xticks([1, 25:25:325, 365]);
    ylabel(['Temperature (', char(176), 'F)']);
    legend('TOBS data (2016)', 'Model');
    M_arr(3) = normalized_misfit(testing_data(:, 7), TOBS_svd(1:365));
end

function [M] = normalized_misfit(wd, w_hat)
    n = length(w_hat);
    norm_arr = zeros(1, n);
    for i = 1:n
       norm_arr = norm(wd - w_hat(i));
    end
    M = norm(norm_arr)/365;
end

function [U, D, V] = calculate_svd(data_vec)
    lmax = 4;
    t = 365;
    c = data_vec(1:(t+lmax));
    r = data_vec((t+lmax):end);
    H = hankel(c, r);
    [U, D, V] = svd(H);
end

function [] = plot_svd(D_PRCP, D_SNOW, D_TOBS)
    diagD_PRCP = diag(D_PRCP(:, 1:365));
    diagD_SNOW = diag(D_SNOW(:, 1:365));
    diagD_TOBS = diag(D_TOBS(:, 1:365));
    days = 1:1:365;
    
    semilogy(days, diagD_PRCP, '-o');
    hold on
    semilogy(days, diagD_SNOW, '-o');
    hold on
    semilogy(days, diagD_TOBS, '-o');
    xlim([0 365])
    xticks([1, 25:25:325, 365])
    yticks([1e-5 1e-4 1e-3 1e-2 1e-1 1, 10, 100, 1000, 1e4, 1e5]);
    xlabel('Day of the year');
    ylabel('SVD values');
    legend('PRCP', 'SNOW', 'TOBS');
    grid on
    grid minor
end

function [H] = calculate_H(ud, H_yd, lmax, t)
    c_ud = ud(1:(lmax+t));
    r_ud = ud((lmax+t):end);
    H_ud = hankel(c_ud, r_ud);

    U_p = H_ud(1:lmax, :);
    U_f = H_ud(lmax+1:end, :);
    Y_p = H_yd(1:lmax, :);
    Y_f = H_yd(lmax+1:end, :);

    a = [U_p; U_f; Y_p];
    b = [zeros(lmax, 1); 1; zeros(t-1, 1); zeros(lmax, 1)];
    G_bar = a\b;
    H = Y_f*G_bar;
end

function [Gamma, A, B, C, D] = realize(H, lmax)
    c = H(1:lmax);
    r = H(lmax:end);
	[U, D, V] = svd(hankel(c, r));
	Gamma = U*sqrtm(D(:, 1:lmax));
	Delta = sqrtm(D(:, 1:lmax))*transpose(V(:, 1:lmax));
	D = H(1);
	C = Gamma(1,:);
	B = Delta(:,1);
	%[m, n] = size(Gamma);
	Gamma_shifted = Gamma(2:end, :);
	A = Gamma_shifted\Gamma(1:end-1, :);
end

function [] = predict_weather()
	fprintf('\n*===============================================================*\n');
	fprintf('*                       WEATHER PREDICTOR                       *\n');
	fprintf('*                          (Boston, MA)                         *');
	fprintf('\n*===============================================================*\n');
	month = input('Enter month (numbers 1 thru 12): ');
	while (month < 1) || (month > 12)
		disp('Invalid month!');
		month = input('Please enter month again (numbers 1 thru 12): ');
	end

	day = input('Enter day: ');
	while (1)
		if ~((day>31) || ((month==4 || month==6 || month==9 || month==11) && day>30) || (month==2 && day>28))
			break
		end
		disp('Invalid date!');
		day = input('Please enter day again: ');
	end
	station = input('Enter station name: ', 's');

	fprintf('\nReading weather data...\n');
	[past_Reading, present_Reading, past_BlueHill, present_BlueHill, past_Bridgewater, present_Bridgewater, past_Groveland, present_Groveland, past_Taunton, present_Taunton, past_Beverly, present_Beverly, past_Norton, present_Norton] = read_data();
	disp('Building prediction model...');

	PRCP = []; SNWD = []; SNOW = []; TMAX = []; TMIN = []; TOBS = [];
	if strcmp(station,'Reading')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_Reading);
	elseif strcmp(station,'Blue Hill')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_BlueHill);
	elseif strcmp(station,'Bridgewater')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_Bridgewater);
	elseif strcmp(station,'Groveland')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_Groveland);
	elseif strcmp(station,'Taunton')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_Taunton);
	elseif strcmp(station,'Beverly')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_Beverly);
	elseif strcmp(station,'Norton')
		[PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(past_Norton);
	end

	fprintf('\n*****************************************************************\n');
	disp([station, ' Station on ', num2str(month), '/', num2str(day), ':']);
	fprintf('\n');
	date_index = date_converter(month, day);
	disp(['Daily average: ', num2str(TOBS(date_index)), char(176), 'F']);
	disp(['Daily high: ', num2str(TMAX(date_index)), char(176), 'F']);
	disp(['Daily low: ', num2str(TMIN(date_index)), char(176), 'F']);
	disp(['Rain: ', num2str(PRCP(date_index)*10), ' mm']);
	disp(['Snow depth: ', num2str(SNWD(date_index)), ' mm']);
	disp(['Snowfall: ', num2str(SNOW(date_index)), ' mm']);
end

function [date_index] = date_converter(month, day)
	num_days_arr = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
	date_index = day;
	if month > 1
		date_index = date_index + sum(num_days_arr(1:(month-1)));
	end
end

function [M_matrix] = find_optimal_lag(training_data, testing_data)
    % setting ident parameters
    m = 0; opt_eiv.exct = [];
    M_matrix = zeros(3, 4);
    ell_arr = 1:4;
    data_index = [2 4 7];
    
    for i = 1:3
        % calculate misfit for different lag
        for ell = ell_arr
            w_mult = zeros(365, 1);
            for k = 1:3
                w_mult(:, :, k) = training_data((365*(k-1)+1):(365*k), data_index(i));
            end
            sys = ident(w_mult, m, ell, opt_eiv);
            M_matrix(i, ell) = misfit(testing_data(:, data_index(i)), sys, opt_eiv);
        end
        % plot misfit vs. lag
        semilogy(M_matrix(i, :), '-o');
        hold on
    end
    
    xticks([1 2 3 4]);
    xticklabels({'1','2','3','4'});
    xlabel('Lag');
    ylabel('Misfit');
    title('Misfit vs. Lag');
    legend('PRCP', 'SNOW', 'TOBS');
    grid on
end

function [PRCP_sys, SNOW_sys, TOBS_sys, M_arr] = plot_ident_sys_model(training_data, testing_data)
    % setting ident parameters
    m = 0; ell = 4; opt_eiv.exct = [];
    M_arr = zeros(1, 3);

    % system identification of PRCP data
    PRCP_mult = zeros(365, 1);
    for k = 1:3
        PRCP_mult(:, :, k) = training_data((365*(k-1)+1):(365*k), 2);
    end
    PRCP_sys = ident(PRCP_mult, m, ell, opt_eiv);
    M_arr(1) = misfit(testing_data(:, 2), PRCP_sys)/365;
    PRCP_sys.TimeUnit = 'days';
    figure
    compare(testing_data(:, 2), PRCP_sys);
    xlabel('Day of the year');
    xticks([1 25:25:325, 365]);
    ylabel('Precipitation (mm)');
    %yticks([0.06 0.08 0.1 0.12]);
    %yticklabels(['1.2', '1.6', '2.0', '2.4']);
    legend('PRCP data (2016)', 'Model'); 
    
    % system identification of SNOW data
    SNOW_mult = zeros(365, 1);
    for k = 1:3
        SNOW_mult(:, :, k) = training_data((365*(k-1)+1):(365*k), 4);
    end
    SNOW_sys = ident(SNOW_mult, m, ell, opt_eiv);
    M_arr(2) = misfit(testing_data(:, 4), SNOW_sys)/365;
    SNOW_sys.TimeUnit = 'days';
    figure
    compare(testing_data(:, 4), SNOW_sys);
    xlabel('Day of the year');
    xticks([1 25:25:325, 365]);
    ylabel('Snow Fall (mm)');
    %yticks([0 0.1 0.2 0.3 0.4 ]);
    %yticklabels(['0', '2', '4', '6', '8']);
    legend('SNOW data (2016)', 'Model');
    
    % system identification of TOBS data
    TOBS_mult = zeros(365, 1);
    for k = 1:3
        TOBS_mult(:, :, k) = training_data((365*(k-1)+1):(365*k), 7);
    end
    TOBS_sys = ident(TOBS_mult, m, ell, opt_eiv);
    M_arr(3) = misfit(testing_data(:, 7), TOBS_sys)/(15*365);
    TOBS_sys.TimeUnit = 'days';
    figure
    compare(testing_data(:, 7), TOBS_sys);
    xlabel('Day of the year');
    xticks([1 25:25:325, 365]);
    ylabel(['Average Daily Observed Temperature (', char(176), 'F)']);
    legend('TOBS data (2016)', 'Model');
end

function [M_arr] = plot_ident_ver2_sys_model(ver2_training_data, testing_data)
    % setting ident parameters
    m = 0; ell = 3; opt_eiv.exct = [];
    M_arr = zeros(1, 3);
    len = length(ver2_training_data(:, 1))/3;
    
    % system identification of PRCP data
    PRCP_mult = zeros(len, 1);
    for k = 1:3
        PRCP_mult(:, :, k) = ver2_training_data((len*(k-1)+1):(len*k), 2);
    end
    PRCP_sys = ident(PRCP_mult, m, ell, opt_eiv);
    M_arr(1) = misfit(testing_data(:, 2), PRCP_sys)/365;
    PRCP_sys.TimeUnit = 'days';
    figure
    compare(testing_data(:, 2), PRCP_sys);
    xlabel('Day of the year');
    xticks([1 25:25:325, 365]);
    ylabel('Precipitation (mm)');
    legend('PRCP data (2016)', 'Model'); 
    
    % system identification of SNOW data
    SNOW_mult = zeros(len, 1);
    for k = 1:3
        SNOW_mult(:, :, k) = ver2_training_data((len*(k-1)+1):(len*k), 4);
    end
    SNOW_sys = ident(SNOW_mult, m, ell, opt_eiv);
    M_arr(2) = misfit(testing_data(:, 4), SNOW_sys)/365;
    SNOW_sys.TimeUnit = 'days';
    figure
    compare(testing_data(:, 4), SNOW_sys);
    xlabel('Day of the year');
    xticks([1 25:25:325, 365]);
    ylabel('Snow Fall (mm)');
    legend('SNOW data (2016)', 'Model');
    
    % system identification of TOBS data
    TOBS_mult = zeros(len, 1);
    for k = 1:3
        TOBS_mult(:, :, k) = ver2_training_data((len*(k-1)+1):(len*k), 7);
    end
    TOBS_sys = ident(TOBS_mult, m, ell, opt_eiv);
    M_arr(3) = misfit(testing_data(:, 7), TOBS_sys)/(15*365);
    TOBS_sys.TimeUnit = 'days';
    figure
    compare(testing_data(:, 7), TOBS_sys);
    xlabel('Day of the year');
    xticks([1 25:25:325, 365]);
    ylabel(['Average Daily Observed Temperature (', char(176), 'F)']);
    legend('TOBS data (2016)', 'Model');
end

function [b] = veronese2(a)
    b = [];
    b_count = 1;
    for i = 1:length(a)
        b(b_count) = a(i)^2;
        b_count = b_count + 1;
        if i < length(a)
            b(b_count) = a(i)*a(i+1);
            b_count = b_count + 1;
        end
    end
end

function [ver2_data] = generate_veronese2_data(data)
    ver2_data = [];
    for i = 2:7
        yr1 = veronese2(data(1:365, i));
        yr2 = veronese2(data(366:2*365, i));
        yr3 = veronese2(data((2*365+1):3*365, i));
        ver2_data(:, i) = [yr1'; yr2'; yr3'];
    end
end

function [stat_table] = calculate_statistics(data)
    % PRCP statistics
    PRCP_diff = abs(data(2:end, 2) - data(1:end-1, 2));
    mean_PRCP_diff = mean(PRCP_diff);
    std_PRCP_diff = std(PRCP_diff);
    max_PRCP_diff = max(PRCP_diff);
    min_PRCP_diff = min(PRCP_diff);

    % SNOW statistics
    SNOW_diff = abs(data(2:end, 4) - data(1:end-1, 4));
    mean_SNOW_diff = mean(SNOW_diff);
    std_SNOW_diff = std(SNOW_diff);
    max_SNOW_diff = max(SNOW_diff);
    min_SNOW_diff = min(SNOW_diff);

    % TOBS statistics
    TOBS_diff = abs(data(2:end, 7) - data(1:end-1, 7));
    mean_TOBS_diff = mean(TOBS_diff);
    std_TOBS_diff = std(TOBS_diff);
    max_TOBS_diff = max(TOBS_diff);
    min_TOBS_diff = min(TOBS_diff);
    
    stat_table = [mean_PRCP_diff mean_SNOW_diff mean_TOBS_diff; 
                  std_PRCP_diff std_SNOW_diff std_TOBS_diff;
                  max_PRCP_diff max_SNOW_diff max_TOBS_diff;
                  min_PRCP_diff min_SNOW_diff min_TOBS_diff;];
end

function [PRCP_sys, SNOW_sys, TOBS_sys] = main()
    disp('Read weather data.');
	[training_data, testing_data] = read_data();
	
%     disp('Find optimal lag.');
%     [M_ident_matrix] = find_optimal_lag(training_data, testing_data);
    
    disp('Model system with ident package.');
    [PRCP_sys, SNOW_sys, TOBS_sys, M_arr_ident] = plot_ident_sys_model(training_data, testing_data);
    
%     disp('Model system with Hankel matrix SVD method.');
%     M_arr_svd = ident_svd(training_data, testing_data);
%     
%     disp('Model system with ident and Veronese embedding of degree 2');
%     ver2_training_data = generate_veronese2_data(training_data);
%     M_arr_ver2 = plot_ident_ver2_sys_model(ver2_training_data, testing_data);
%     
%     disp('Calculate weather statistics.');
%     [stat_table] = calculate_statistics([training_data; testing_data]);
end
