%% Load and Reshape Lidar Data
% File paths
binFile = "C:\TexLiDAR\Images\8344.bin";
ambientFile = "C:\TexLiDAR\Images\8344.png";
% Read binary data
fileID = fopen(binFile, 'rb');
lidarData = fread(fileID, [4, Inf], 'single')';
fclose(fileID);

% Extract point attributes
[x, y, z, intensity] = deal(lidarData(:, 1), lidarData(:, 2), lidarData(:, 3), lidarData(:, 4));

% Image dimensions
imgWidth = 2048;
imgHeight = 128;

% Reshape attributes to images
intensityImg = reshape(intensity, [imgWidth, imgHeight])';
xImg = reshape(x, [imgWidth, imgHeight])';
yImg = reshape(y, [imgWidth, imgHeight])';
zImg = reshape(z, [imgWidth, imgHeight])';


%% Define ROI
roi = [71.93600463867188, 63.552001953125, 99.5840072631836, 109.24800872802734]; % [x1, y1, x2, y2]
direction = 'front';
[x_min_full, y_min, x_max_full, y_max] = adjust_roi_in_full_image(roi, direction);
% If x_min_full > x_max_full, calculate the midpoint with wraparound
if x_min_full > x_max_full
    roi_length = IMAGE_WIDTH - x_min_full + x_max_full;
    x_center = mod(x_min_full + floor(roi_length / 2), imgWidth);
else
    % Otherwise, calculate the midpoint normally
    x_center = floor((x_min_full + x_max_full) / 2);
end
% Calculate the center for the y-axis (no wraparound)
 y_center = floor((y_min + y_max) / 2);
 
%% Calculate Distance and Angle
% Extract midpoint coordinates
[xMid, yMid, zMid] = deal(xImg(y_center, x_center), yImg(y_center, x_center), zImg(y_center, x_center));

% Compute distance and angle
distance = sqrt(xMid^2 + yMid^2);
angle = 360 * (x_center - imgWidth/2) / imgWidth;

% Display results
disp(['Distance: ', num2str(distance)]);
disp(['Angle (degrees): ', num2str(angle)]);
%%
function [x_min_full, y_min, x_max_full, y_max] = adjust_roi_in_full_image(roi, direction)
    % Adjust the given ROI for a specific direction in the full image based on direction.
    %
    % Parameters:
    % - roi: Vector [x_min, y_min, x_max, y_max] for the ROI in the cropped slice.
    % - direction: One of {'left', 'front', 'right', 'back'} indicating the direction.
    %
    % Returns:
    % - Vector [x_min_full, y_min, x_max_full, y_max] for the adjusted ROI in the full image.

    % Define constants (assuming these are predefined elsewhere or passed in)
    
    % Map direction to slice index
    CROP_WIDTH = 512; IMAGE_WIDTH= 2048; OFFSET = 256;
    direction_map = containers.Map({'left', 'front', 'right', 'back'}, {1, 2, 3, 4});

    % Ensure the direction is valid
    if ~isKey(direction_map, direction)
        error('Invalid direction. Choose from {"left", "front", "right", "back"}.');
    end

    % Get the slice index for the given direction
    index = direction_map(direction);

    % Calculate start column for the slice in the full image
    start_col = (index - 1) * CROP_WIDTH + OFFSET;

    % Extract ROI components
    x_min_crop = roi(1);
    y_min = roi(2);
    x_max_crop = roi(3);
    y_max = roi(4);

    % Adjust the x-coordinates in the ROI for the full image
    x_min_full = mod(x_min_crop + start_col, IMAGE_WIDTH);
    x_max_full = mod(x_max_crop + start_col, IMAGE_WIDTH);

    % Return the adjusted ROI
    return
end
