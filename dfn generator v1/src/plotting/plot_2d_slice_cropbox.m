function plot_2d_slice_cropbox(masterFile, cropBox, sliceAxis, sliceVal)
% PLOT_2D_SLICE_CROPBOX Generates a 2D trace map of the DFN on a specified plane.
% 
% masterFile : Path to the dfn_master_index.mat
% cropBox    : Crop box structure (xmin, xmax, ymin, ymax, zmin, zmax)
% sliceAxis  : String 'x', 'y', or 'z' specifying the slice plane
% sliceVal   : Normal coordinate value of the slicing plane

    M = load(masterFile);
    master = M.master;
    
    sliceAxis = lower(sliceAxis);
    dim = double(sliceAxis) - 'x' + 1; % 'x'=1, 'y'=2, 'z'=3
    
    figure('Color','w','Position',[200 200 800 800]);
    hold on; axis equal; grid on;
    
    if dim == 1
        xlabel('Y [m]'); ylabel('Z [m]'); title(sprintf('2D Trace Map at X = %.2f', sliceVal));
        axis([cropBox.ymin cropBox.ymax cropBox.zmin cropBox.zmax]);
        idx1 = 2; idx2 = 3;
    elseif dim == 2
        xlabel('X [m]'); ylabel('Z [m]'); title(sprintf('2D Trace Map at Y = %.2f', sliceVal));
        axis([cropBox.xmin cropBox.xmax cropBox.zmin cropBox.zmax]);
        idx1 = 1; idx2 = 3;
    else
        xlabel('X [m]'); ylabel('Y [m]'); title(sprintf('2D Trace Map at Z = %.2f', sliceVal));
        axis([cropBox.xmin cropBox.xmax cropBox.ymin cropBox.ymax]);
        idx1 = 1; idx2 = 2;
    end
    
    % Plot the cropbox boundary acting as the viewing window
    bw = [cropBox.xmin, cropBox.xmax, cropBox.ymin, cropBox.ymax, cropBox.zmin, cropBox.zmax];
    rect_x = [bw(idx1*2-1) bw(idx1*2) bw(idx1*2) bw(idx1*2-1) bw(idx1*2-1)];
    rect_y = [bw(idx2*2-1) bw(idx2*2-1) bw(idx2*2) bw(idx2*2) bw(idx2*2-1)];
    plot(rect_x, rect_y, 'k-', 'LineWidth', 2);
    
    total_trace_len = 0;
    
    for k = 1:numel(master.set_files)
        S = load(master.set_files{k});
        centers = double(S.centers);
        normals = double(S.normals);
        radius  = double(S.radius);
        
        % 1. AABB Cull (Only fractures physically touching the cropbox)
        valid_mask = ...
            (centers(:,1) - radius <= cropBox.xmax) & (centers(:,1) + radius >= cropBox.xmin) & ...
            (centers(:,2) - radius <= cropBox.ymax) & (centers(:,2) + radius >= cropBox.ymin) & ...
            (centers(:,3) - radius <= cropBox.zmax) & (centers(:,3) + radius >= cropBox.zmin);
        
        % 2. Plane cull (Must cross the slice plane)
        valid_mask = valid_mask & (centers(:,dim) - radius <= sliceVal) & (centers(:,dim) + radius >= sliceVal);
        
        valid_idx = find(valid_mask);
        nValid = length(valid_idx);
        
        segs_ch = cell(nValid, 1);
        
        % Parallel generation
        parfor idx = 1:nValid
            i = valid_idx(idx);
            
            % Generate the 3D polygon inside the cropbox first
            poly = clip_disc_with_cropbox(centers(i,:), normals(i,:), radius(i), cropBox);
            
            if ~isempty(poly)
                % Then slice the 3D polygon with the 2D plane to get a line trace
                seg = intersect_poly_plane(poly, dim, sliceVal);
                segs_ch{idx} = seg;
            end
        end
        
        % Plot lines sequentially
        for idx = 1:nValid
            seg = segs_ch{idx};
            if ~isempty(seg)
                plot(seg(:,idx1), seg(:,idx2), '-', 'Color', 'k', 'LineWidth', 1.0);
                total_trace_len = total_trace_len + norm(seg(1,:) - seg(2,:));
            end
        end
    end
    
    % Report P21 Areal Intensity
    A_box = (bw(idx1*2) - bw(idx1*2-1)) * (bw(idx2*2) - bw(idx2*2-1));
    p21 = total_trace_len / A_box;
    
    fprintf('\n--- 2D Slice Report ---\n');
    fprintf('Slice %s = %.2f m\n', upper(sliceAxis), sliceVal);
    fprintf('Total Trace Length = %.4f m\n', total_trace_len);
    fprintf('Observation Area = %.4f m^2\n', A_box);
    fprintf('P21 (Areal Intensity) = %.4f m^-1\n', p21);
    fprintf('-----------------------\n\n');
end

function seg = intersect_poly_plane(poly, dim, val)
% Find the line segment where a 3D convex polygon intersects an axis-aligned plane
    tol = 1e-8;
    pts = [];
    N = size(poly,1);
    for i = 1:N
        P1 = poly(i,:);
        P2 = poly(mod(i,N)+1,:);
        
        v1 = P1(dim);
        v2 = P2(dim);
        
        if abs(v1 - val) < tol
            pts = [pts; P1];
        end
        
        if (v1 - val) * (v2 - val) < 0
            t = (val - v1) / (v2 - v1);
            I = P1 + t * (P2 - P1);
            pts = [pts; I];
        end
        
        if abs(v1 - val) < tol && abs(v2 - val) < tol
            pts = [pts; P2];
        end
    end
    
    if isempty(pts)
        seg = []; return;
    end
    
    pts = uniquetol(pts, 1e-7, 'ByRows', true);
    if size(pts,1) < 2
        seg = []; return;
    end
    
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
