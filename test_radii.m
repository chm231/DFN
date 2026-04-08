rmin = 1.0;
rmax = 250.0;
kr = 2.88;
N = 2400000;
U = rand(N, 1, 'single');
radius_ch = rmin * ( U .* ( (rmin/rmax)^kr - 1 ) + 1 ).^(-1/kr);
mean_area = mean(pi * double(radius_ch).^2);
disp(['Empirical Mean Area using U (single) = ', num2str(mean_area)]);

U2 = rand(N, 1, 'double');
radius_ch2 = rmin * ( U2 .* ( (rmin/rmax)^kr - 1 ) + 1 ).^(-1/kr);
mean_area2 = mean(pi * radius_ch2.^2);
disp(['Empirical Mean Area using U (double) = ', num2str(mean_area2)]);
