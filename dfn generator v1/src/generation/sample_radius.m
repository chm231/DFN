function r = sample_radius(sizeDist, N)
% Sample truncated fracture radius.

    rmin = sizeDist.rmin;
    rmax = sizeDist.rmax;
    U = rand(N,1);

    switch lower(sizeDist.type)

        case 'powerlaw'
            k = sizeDist.kr;

            if abs(k - 1) < 1e-12
                r = rmin .* (rmax/rmin).^U;
            else
                r = ( rmin^(1-k) + U .* (rmax^(1-k) - rmin^(1-k)) ).^(1/(1-k));
            end

        case 'exponential'
            lambda = 1 / sizeDist.r0;
            A = exp(-lambda*rmin);
            B = exp(-lambda*rmax);
            r = -(1/lambda) .* log(A - U .* (A - B));

        case 'lognormal'
            mu    = sizeDist.mu;
            sigma = sizeDist.sigma;

            r = zeros(N,1);
            cnt = 0;

            while cnt < N
                batch = lognrnd(mu, sigma, [N,1]);
                batch = batch(batch >= rmin & batch <= rmax);
                take = min(numel(batch), N-cnt);

                if take > 0
                    r(cnt+1:cnt+take) = batch(1:take);
                    cnt = cnt + take;
                end
            end

        case 'uniform'
            r = rmin + (rmax-rmin).*U;

        otherwise
            error('Unknown sizeDist.type: %s', sizeDist.type);
    end
end