% add path for the SLRA packet
addpath(fullfile(cd, 'slra-slra-c3aa24c'));
warning('off', 'all');
[training_data, testing_data, w_mult, M_arr] = test_fun();

% read weather data from the file
function [training_data, testing_data] = read_data()
    data_Reading = csvread('962598.csv', 70136, 2, [70136, 2, 72082, 13]);

    % remove unnecessary columns
    % columns: DATE, PRCP, SNWD, SNOW, TMAX, TMIN, TOBS
    cols2remove = [2 3 7 8 9];
    data_Reading(:, cols2remove) = [];

    % remove 2/29/2012 from the data set
    data_Reading(60, :) = [];

    % remove invalid values in the data
    data_Reading = preprocess_data(data_Reading);

    % filter noise
    data_Reading = filter_noise(data_Reading);
    
    % split data into training set (2013-2015) and testing set (2016)
    [training_data, testing_data] = split_data(data_Reading);
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
    daysPerMonth = 30;
    coeff30dMA = ones(1, daysPerMonth)/daysPerMonth;
    avg30dPRCP = filter(coeff30dMA, 1, data(:, 7));
    avg30dSNWD = filter(coeff30dMA, 1, data(:, 7));
    avg30dSNOW = filter(coeff30dMA, 1, data(:, 7));
    avg30dTMAX = filter(coeff30dMA, 1, data(:, 7));
    avg30dTMIN = filter(coeff30dMA, 1, data(:, 7));
    avg30dTOBS = filter(coeff30dMA, 1, data(:, 7));
    data = [data(:, 1), avg30dPRCP, avg30dSNWD, avg30dSNOW, avg30dTMAX, avg30dTMIN, avg30dTOBS];
end

function [training_data, testing_data] = split_data(data)
	training_data = data(366:4*365, :);
	testing_data = data((4*365+1):(5*365), :);
end

function [H, U, D, V] = single_value_decomposition(data_vec)
	c = data_vec(1:365);
	r = data_vec(365:end);
	H = hankel(c, r);
	[U, D, V] = svd(H);
end

function [H] = impulse_response(ud, yd, lmax, t)
    c_ud = ud(1:(lmax+t));
    r_ud = ud((lmax+t):end);
    H_ud = hankel(c_ud, r_ud);

    c_yd = yd(1:(lmax+t));
    r_yd = yd((lmax+t):end);
    H_yd = hankel(c_yd, r_yd);

    U_p = H_ud(1:lmax, :);
    U_f = H_ud(lmax+1:end, :);
    Y_p = H_yd(1:lmax, :);
    Y_f = H_yd(lmax+1:end, :);

    A = [U_p;U_f;Y_p];
    B = [zeros(lmax, 1); 1; zeros(t-1, 1); zeros(lmax, 1)];
    G_bar = A\B;
    H = Y_f*G_bar;
end

function [Y_0] = zero_input_response(ud, yd, lmax, t)
	c_ud = ud(1:(lmax+t));
    r_ud = ud((lmax+t):end);
    H_ud = hankel(c_ud, r_ud);

    c_yd = yd(1:(lmax+t));
    r_yd = yd((lmax+t):end);
    H_yd = hankel(c_yd, r_yd);

    U_p = H_ud(1:lmax, :);
    U_f = H_ud(lmax+1:end, :);
    Y_p = H_yd(1:lmax, :);
    Y_f = H_yd(lmax+1:end, :);

    A = [U_p;U_f;Y_p];
    [m, n] = size(U_f);
    B = [U_p; zeros(m,n); Y_p];
    G_bar = A\B;
    Y_0 = Y_f*G_bar;
end

function [A, B, C, D] = realization(H, lmax)
	c = [H;0];
	[U, D, V] = svd(hankel(c));
	Gamma = U*sqrtm(D);
	Delta = sqrtm(D)*transpose(V);
	D = H(1);
	C = Gamma(1,:);
	B = Delta(:,1);
	[m, n] = size(Gamma);
	Gamma_shifted = Gamma(2:end, :);;
	A = Gamma_shifted\Gamma(1:lmax, :);
end

function [PRCP, SNWD, SNOW, TMAX, TMIN, TOBS] = exact_model(data)
	lmax = 365;
	t = 365;
	days = 1:1:(5*365+1);
    ud = sin(((days-79.5)/365)*(2*pi));
    H_PRCP = impulse_response(ud, data(:, 2), lmax, t);
    PRCP = conv(H_PRCP, ud);
    PRCP = PRCP(1:365) + mean(data(:, 2));

    H_SNWD = impulse_response(ud, data(:, 3), lmax, t);
    SNWD = conv(H_SNWD, ud);
    SNWD = SNWD(1:365) + mean(data(:, 3));

    H_SNOW = impulse_response(ud, data(:, 4), lmax, t);
    SNOW = conv(H_SNOW, ud);
    SNOW = SNOW(1:365) + mean(data(:, 4));

    H_TMAX = impulse_response(ud, data(:, 5), lmax, t);
    TMAX = conv(H_TMAX, ud);
    TMAX = TMAX(1:365) + mean(data(:, 5));

    H_TMIN = impulse_response(ud, data(:, 6), lmax, t);
    TMIN = conv(H_TMIN, ud);
    TMIN = TMIN(1:365) + mean(data(:, 6));

    H_TOBS = impulse_response(ud, data(:, 7), lmax, t);
    TOBS = conv(H_TOBS, ud);
    TOBS = TOBS(1:365) + mean(data(:, 7));
end

function [inp, data] = slra_model(data)
	fprintf('Building SLRA model.\n');
	m = 1; ell = 2;
	opt_oe.exct = 1:m;
	opt_eiv.exct = [];
	N = 5;

	d = 1:1:(365);
	u0 = transpose(sin((d-79.5)/365)*(2*pi));
	inp = u0;

%	u_mult = {}; y_mult = {}; w_mult = [];
% 	for k = 1:N
% 		y_mult{k} = data((365*(k-1)+1):(365*k), 7);
% 		u_mult{k} = u0;
% 		w_mult(:, :, k) = [u0 y_mult{k}];
% 	end
	% d = 1:1:(365*5+1);
	% u = transpose(sin((d-79.5)/365)*(2*pi));
	% inp = u;

	% yd = data(:, 7)
	% w = [u yd]; 
	%[sysh, info, wh, xini] = ident(w_mult, m, ell, opt_oe);
	%tic, sys = pem(iddata(y_mult, u_mult), 2, 'dist', 'none'); t_pem = toc;
	%sysh2 = ident(fliplr(w), m, ell, opt_eiv);
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

function [w_mult, M_arr] = find_optimal_lag(training_data, testing_data)
    % setting ident parameters
    m = 0; opt_eiv.exct = [];
    M_arr = zeros(1, 4);
    ell_arr = 1:4;
    
    % calculate misfit for different lag
    for ell = ell_arr
        w_mult = zeros(365, 1);
        for k = 1:3
            w_mult(:, :, k) = training_data((365*(k-1)+1):(365*k), 7);
        end
        sys = ident(w_mult, m, ell, opt_eiv);
        M_arr(ell) = misfit(testing_data(:, 7), sys, opt_eiv);
    end

    % plot misfit vs. lag
    plot(M_arr, '-ob');
    xticks([1 2 3 4]);
    xticklabels({'1','2','3','4'});
    xlabel('Lag');
    ylabel('Misfit');
    title('Misfit vs. Lag');
end

function [training_data, testing_data, M_arr] = test_fun()
    disp('Read weather data.');
	[training_data, testing_data] = read_data();
	
    disp('Find optimal lag.');
    [M_arr] = find_optimal_lag(training_data, testing_data);
end
