%%
% Define the input and output folder paths
inputFolderPath = '';
outputFolderPath = '';

% Ensure the output folder exists
if ~exist(outputFolderPath, 'dir')
    mkdir(outputFolderPath);
end

% Define the small image size
smallImageSize = 512; % Width of each small image
shiftAmount = 256;

% Get a list of all PNG files in the input folder
imageFiles = dir(fullfile(inputFolderPath, '*.png'));

% Loop through all images in the folder
for fileIdx = 1:length(imageFiles)
    % Get the full path of the current image
    inputImagePath = fullfile(inputFolderPath, imageFiles(fileIdx).name);
    
    % Read the input image
    inputImage = imread(inputImagePath);
    
    % Get the dimensions of the input image
    [rows, cols, channels] = size(inputImage);
    
    % Check if the image dimensions are valid
    if rows ~= 128 || mod(cols, smallImageSize) ~= 0
        fprintf('Skipping %s: Invalid dimensions.\n', imageFiles(fileIdx).name);
        continue;
    end
    
    % Calculate the number of smaller images to create
    numImages = cols / smallImageSize;
    
    % Process the image and save smaller sections
    for i = 1:numImages
        % Calculate the column indices for the current small image
        colStart = mod((i - 1) * smallImageSize + shiftAmount, cols) + 1;
        colEnd = mod(i * smallImageSize + shiftAmount, cols);
        
        % Extract the patch in two parts if colEnd < colStart (wrap-around)
        if colEnd < colStart
            smallImage1 = inputImage(:, colStart:cols, :);  % From colStart to the end of the image
            smallImage2 = inputImage(:, 1:colEnd, :);      % From the start of the image to colEnd
            smallImage = [smallImage1, smallImage2];        % Concatenate both parts
        else
            % Normal case without wrap-around
            smallImage = inputImage(:, colStart:colEnd, :);
        end
        
        % Define the output image file name
        outputFileName = sprintf('%s_image_%03d.png', imageFiles(fileIdx).name(1:end-4), i);
        
        % Create the full output path
        outputPath = fullfile(outputFolderPath, outputFileName);
        
        % Save the smaller image
        imwrite(smallImage, outputPath);
    end
    
    fprintf('Processed %s: Saved %d small images.\n', imageFiles(fileIdx).name, numImages);
end

disp('All images processed successfully.');
