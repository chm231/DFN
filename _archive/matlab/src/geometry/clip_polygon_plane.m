function poly_out = clip_polygon_plane(poly_in, n, d)
% Keep region satisfying: n·x + d >= 0

    if isempty(poly_in)
        poly_out = [];
        return;
    end

    poly_out = [];

    N = size(poly_in,1);

    for i = 1:N

        P1 = poly_in(i,:);
        P2 = poly_in(mod(i,N)+1,:);

        val1 = dot(n, P1) + d;
        val2 = dot(n, P2) + d;

        inside1 = val1 >= 0;
        inside2 = val2 >= 0;

        if inside1 && inside2
            % both inside → keep P2
            poly_out = [poly_out; P2];

        elseif inside1 && ~inside2
            % leaving → add intersection
            t = val1 / (val1 - val2);
            I = P1 + t*(P2 - P1);
            poly_out = [poly_out; I];

        elseif ~inside1 && inside2
            % entering → add intersection + P2
            t = val1 / (val1 - val2);
            I = P1 + t*(P2 - P1);
            poly_out = [poly_out; I; P2];

        else
            % both outside → discard
        end
    end
end