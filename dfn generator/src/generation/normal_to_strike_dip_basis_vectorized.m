function [strike_u, dip_u] = normal_to_strike_dip_basis_vectorized(normals)
% Build two orthonormal in-plane vectors from normals.

    cls = class(normals);

    nrm = sqrt(sum(normals.^2, 2));
    normals = normals ./ nrm;

    N = size(normals,1);

    ref = zeros(N,3, cls);
    ref(:,3) = 1;

    mask = abs(normals(:,3)) > 0.95;
    ref(mask,:) = 0;
    ref(mask,1) = 1;

    strike_u = cross(ref, normals, 2);
    strike_u = strike_u ./ sqrt(sum(strike_u.^2, 2));

    dip_u = cross(normals, strike_u, 2);
    dip_u = dip_u ./ sqrt(sum(dip_u.^2, 2));
end