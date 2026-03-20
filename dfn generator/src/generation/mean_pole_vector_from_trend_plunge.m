function n = mean_pole_vector_from_trend_plunge(trend_deg, plunge_deg)
% x = East, y = North, z = Up
% plunge positive downward

    tr = deg2rad(trend_deg);
    pl = deg2rad(plunge_deg);

    n = [cos(pl)*sin(tr), cos(pl)*cos(tr), -sin(pl)];
    n = n / norm(n);
end