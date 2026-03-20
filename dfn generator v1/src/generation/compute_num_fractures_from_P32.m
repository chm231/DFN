function N = compute_num_fractures_from_P32(P32, sizeDist, V)
% Compute fracture count from truncated P32 in the simulated radius range:
%   N = V * P32 / E[pi r^2]
%
% Here P32 is interpreted consistently over [rmin, rmax].

    rmin = sizeDist.rmin;
    rmax = sizeDist.rmax;

    pdf = @(r) size_pdf_truncated(r, sizeDist);
    meanArea = integral(@(r) pi .* r.^2 .* pdf(r), rmin, rmax, 'ArrayValued', true);

    N_real = V * P32 / meanArea;
    N = max(0, round(N_real));
end

function val = size_pdf_truncated(r, sizeDist)
    rmin = sizeDist.rmin;
    rmax = sizeDist.rmax;

    val = zeros(size(r));
    mask = (r >= rmin) & (r <= rmax);

    switch lower(sizeDist.type)
        case 'powerlaw'
            k = sizeDist.kr;

            if abs(k - 1) < 1e-12
                C = 1 / log(rmax / rmin);
            else
                C = (1-k) / (rmax^(1-k) - rmin^(1-k));
            end

            val(mask) = C .* r(mask).^(-k);

        case 'exponential'
            lambda = 1 / sizeDist.r0;
            C = lambda / (exp(-lambda*rmin) - exp(-lambda*rmax));
            val(mask) = C .* exp(-lambda .* r(mask));

        case 'lognormal'
            mu    = sizeDist.mu;
            sigma = sizeDist.sigma;
            raw = @(x) (1 ./ (x*sigma*sqrt(2*pi))) .* exp(-(log(x)-mu).^2 ./ (2*sigma^2));
            Z = integral(raw, rmin, rmax, 'ArrayValued', true);
            val(mask) = raw(r(mask)) ./ Z;

        case 'uniform'
            C = 1 / (rmax - rmin);
            val(mask) = C;

        otherwise
            error('Unknown sizeDist.type: %s', sizeDist.type);
    end
end