function plot_segment_inside_horseshoe(seg, tunnel)
% seg : 2x3, x는 일정, yz 평면에서 선분
% tunnel 내부 부분만 샘플링해서 plot

    y1 = seg(1,2); z1 = seg(1,3);
    y2 = seg(2,2); z2 = seg(2,3);

    t = linspace(0,1,300);
    y = y1 + t*(y2 - y1);
    z = z1 + t*(z2 - z1);

    inside = in_horseshoe_yz(y, z, tunnel);

    if ~any(inside)
        return;
    end

    % inside인 구간을 연속 구간으로 나눠서 그림
    d = diff([false inside false]);
    iStart = find(d == 1);
    iEnd   = find(d == -1) - 1;

    for k = 1:numel(iStart)
        idx = iStart(k):iEnd(k);
        plot(y(idx), z(idx), 'r-', 'LineWidth', 1.8);
    end
end