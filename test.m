% addpath(fullfile(cd, 'slra-slra-c3aa24c'));
% warning('off', 'all');
% 
% data_Reading = csvread('962598.csv', 70136, 2, [70136, 2, 72082, 13]);
% 
% cols2remove = [2 3 7 8 9];
% data_Reading(:, cols2remove) = [];
% 
% data_Reading(60, :) = [];
% 
% data_Reading = preprocess_data(data_Reading);
% 
% daysPerSeason = 90;
% coeff90dMA = ones(1, daysPerSeason)/daysPerSeason;
% avg90dPRCP = filter(coeff90dMA, 1, data_Reading(:, 4));
% coeff30dMA = ones(1, 30)/30;
% avg90dPRCP = filter(coeff30dMA, 1, avg90dPRCP);
% avg30dTOBS = filter(coeff30dMA, 1, data_Reading(:, 7));
% 
% training_data = avg30dTOBS(366:4*365);
% testing_data = avg30dTOBS((4*365+1):(5*365));
% 
% m = 0; ell = 4; opt_eiv.exct = [];
% w_mult = zeros(365, 1);
% for k = 1:3
%     w_mult(:, :, k) = training_data((365*(k-1)+1):(365*k));
% end
% sys = ident(w_mult, m, ell, opt_eiv);
% compare(testing_data(), sys);
% 
% function [data] = preprocess_data(data)
% 	for i = 1:size(data, 1)
% 		% Process invalid PRCP, SNWD, SNOW values
% 		if data(i, 2) == -9999
% 			data(i, 2) = 0;
% 		end
% 		if data(i, 3) == -9999
% 			data(i, 3) = 0;
% 		end
% 		if data(i, 4) == -9999
% 			data(i, 4) = 0;
% 		end
% 
% 		% Process invalid TMAX values
% 		if (data(i, 5) == -9999)
% 			if ((data(i, 6) ~= -9999) && (data(i, 7) ~= -9999))
% 				data(i, 5) = 2*data(i, 7) - data(i, 6);
% 			elseif (i < 365)
% 				data(i, 5) = data(i+365, 5);
% 			else
% 				data(i, 5) = data(i-365, 5);
% 			end
% 		end
% 
% 		% Process invalid TMIN values
% 		if (data(i, 6) == -9999)
% 			if ((data(i, 5) ~= -9999) && (data(i, 7) ~= -9999))
% 				data(i, 6) = data(i, 5) - 2*data(i, 7);
% 			elseif (i < 365)
% 				data(i, 6) = data(i+365, 6);
% 			else
% 				data(i, 6) = data(i-365, 6);
% 			end
% 		end
% 
% 		% Process invalid TOBS values
% 		if (data(i, 7) == -9999)
% 			if ((data(i, 5) ~= -9999) && (data(i, 6) ~= -9999))
% 				data(i, 7) = (data(i, 5)+data(i, 6))/2;
% 			elseif (i < 365)
% 				data(i, 7) = data(i+365, 7);
% 			else
% 				data(i, 7) = data(i-365, 7);
% 			end
% 		end
% 	end
% end

a = [1 2 3 4];
b = veronese2(a);

function [b] = veronese2(a)
    b = [];
    b_count = 1;
    for i = 1:length(a)
        b(b_count) = a(i)^2;
        b_count = b_count + 1;
        if i < length(a)
            b(b_count) = a(i)*a(i+1);
            b_count = b_count + 1;
            if i < length(a)-1
                b(b_count) = a(i)*a(i+2);
                b_count = b_count + 1;
            end
        end
    end
end
