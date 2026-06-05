function seg = intersect_polygon_with_xplane(poly, xFace)
% poly : Nx3 ordered polygon vertices
% seg  : 2x3 line segment on plane x = xFace
%
% 빈 배열이면 교차 없음

    tol = 1e-8;
    pts = [];

    N = size(poly,1);

    for i = 1:N
        P1 = poly(i,:);
        P2 = poly(mod(i,N)+1,:);

        x1 = P1(1);
        x2 = P2(1);

        % case 1: endpoint on plane
        if abs(x1 - xFace) < tol
            pts = [pts; P1];
        end

        % case 2: edge crosses plane
        if (x1 - xFace) * (x2 - xFace) < 0
            t = (xFace - x1) / (x2 - x1);
            I = P1 + t * (P2 - P1);
            pts = [pts; I];
        end

        % case 3: whole edge on plane
        if abs(x1 - xFace) < tol && abs(x2 - xFace) < tol
            pts = [pts; P2];
        end
    end

    if isempty(pts)
        seg = [];
        return;
    end

    % 중복 제거
    pts = uniquetol(pts, 1e-7, 'ByRows', true);

    if size(pts,1) < 2
        seg = [];
        return;
    end

    % 점이 2개 초과일 때는 가장 멀리 떨어진 두 점 선택
    maxd = -inf;
    seg = [];

    for i = 1:size(pts,1)
        for j = i+1:size(pts,1)
            d = norm(pts(i,:) - pts(j,:));
            if d > maxd
                maxd = d;
                seg = [pts(i,:); pts(j,:)];
            end
        end
    end
end