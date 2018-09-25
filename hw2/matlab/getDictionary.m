function [dictionary] = getDictionary(imgPaths, alpha, K, method)
% Generate the filter bank and the dictionary of visual words
% Inputs:
%   imgPaths:        array of strings that repesent paths to the images
%   alpha:          num of points
%   K:              K means parameters
%   method:         string 'random' or 'harris'
% Outputs:
%   dictionary:         a length(imgPaths) * K matrix where each column
%                       represents a single visual word
    % -----fill in your implementation here --------
    % create filter bank
    filterBank = createFilterBank();
    pixelResponses = zeros(alpha*length(imgPaths),3*length(filterBank));
    k = 0.05;  % param for harris points
    for i = 1 : length(imgPaths)
        I = imread(['../data/' imgPaths{1,i}]);    
        % rgb to grayscale
        if ndims(I) == 3
            I_gray = rgb2gray(I);
        else
            I_gray = I;
        end
        
        % get filterResponses
        [h, w] = size(I_gray);
        filterResponses = extractFilterResponses(I, filterBank);
        filterResponses = reshape(filterResponses, h*w, size(filterResponses, 3));
        
        
        % get points position
        if method == 'Harris'
            points = getHarrisPoints(I_gray, alpha, k);
        else
            points = getRandomPoints(I_gray, alpha);
        end
        points_flatten = zeros(size(points,1), 1);
        points_flatten = (points(:,2)-1) * h + points(:,1);
        
        % get points filter responses
        pixelResponses(alpha*(i-1)+1:alpha*i, :) = filterResponses(points_flatten,:);
        
        fprintf(['finished ', num2str(i), '\n']);
    end
    
    % kmeans
    [~, dictionary] = kmeans(pixelResponses, K, 'EmptyAction', 'drop');
    fprintf(['finished method: ', method, '\n']);
        

    % ------------------------------------------
    
end
