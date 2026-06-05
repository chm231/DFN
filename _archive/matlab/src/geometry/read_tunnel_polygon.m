function poly = read_tunnel_polygon(filepath)
% READ_TUNNEL_POLYGON Parses 2D XYZ structured coordinates from the provided .dat file
%
% Usage:
%   poly = read_tunnel_polygon('단면_폴리곤.dat')
%
% This function reads lines formatted as `X=... Y=... Z=...` 
% Assuming inputs are in millimeters, it normalizes and converts them 
% explicitly into meters [m] for seamless Boolean intersection with the DFN.
%
% Returns:
%   poly : [N x 2] array of metric (x, y) coordinates defining the closed horseshoe.

    fid = fopen(filepath, 'rt');
    if fid == -1
        error('Cannot open file: %s', filepath);
    end
    
    x_coords = [];
    y_coords = [];
    
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        
        % Skip empty lines or pure comments (excluding coordinate comments)
        if isempty(line) || startsWith(line, '#')
            continue;
        end
        
        % Target the exact precision format lines: "X=5038.262  Y= 0.000  Z= 0.000"
        if startsWith(line, 'X=')
            % Capture sequences using standard regular expressions 
            % matching any digits, periods, or negative signs mapping to X and Y
            tokens = regexp(line, 'X=\s*([-\d.]+)\s*Y=\s*([-\d.]+)', 'tokens');
            if ~isempty(tokens)
                x_coords(end+1, 1) = str2double(tokens{1}{1});
                y_coords(end+1, 1) = str2double(tokens{1}{2});
            end
        end
    end
    fclose(fid);
    
    if isempty(x_coords)
        error('No valid X=... Y=... coordinates found during parsing.');
    end
    
    % DFN domain generates and operates fundamentally on scale: [Meters]
    % Original coordinates are explicitly logged in Millimeters [mm].
    poly = [x_coords, y_coords] / 1000.0;
end
