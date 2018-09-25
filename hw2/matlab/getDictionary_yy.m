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
    k = 0.05;
    [filterBank] = createFilterBank();
%     load('../data/traintest.mat'); 
    pixelResponses = zeros(alpha*length(imgPaths),3*length(filterBank));
    for i = 1:length(imgPaths)
        fprintf('creating responses from image %d\n',i);
        img = imgPaths(i);
        I = imread(['../data/' img{:}]);
        [filterResponses] = extractFilterResponses(I, filterBank);
        if ndims(I) == 3
            I_gray = rgb2gray(I);
        else
            I_gray = I;
        end
        if method =='random'
            [points] = getRandomPoints(I_gray, alpha);
        end
        if method == 'harris'
            [points] = getHarrisPoints(I_gray, alpha, k);
        end
        pixelResponses(alpha*(i-1)+1:alpha*i,:) = filterResponses(points,:);
    end
    fprintf('finishing pixelResponses');
    [~, dictionary] = kmeans(pixelResponses, K, 'EmptyAction','drop');
    % ------------------------------------------
    
end
