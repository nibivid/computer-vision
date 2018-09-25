function [wordMap] = getVisualWords(I, dictionary, filterBank)
% Convert an RGB or grayscale image to a visual words representation, with each
% pixel converted to a single integer label.   
% Inputs:
%   I:              RGB or grayscale image of size H * W * C
%   filterBank:     cell array of matrix filters used to make the visual words.
%                   generated from getFilterBankAndDictionary.m
%   dictionary:     matrix of size 3*length(filterBank) * K representing the
%                   visual words computed by getFilterBankAndDictionary.m
% Outputs:
%   wordMap:        a matrix of size H * W with integer entries between
%                   1 and K

    % -----fill in your implementation here --------
    
    % get filter response
    [h, w, ~] = size(I);
    filterResponses = extractFilterResponses(I, filterBank);
    filterResponses = reshape(filterResponses, h*w, size(filterResponses, 3));
    
    % get distance to cluster center
    % res=H*W,60, dic=100,60, dis=H*W,100
    distance = pdist2(filterResponses, dictionary, 'euclidean');
    [distance_sort, idx] = sort(distance, 2);
    idx = idx(:,1);
    
    wordMap = reshape(idx, h, w);
    imshow(label2rgb(wordMap));
    % ------------------------------------------
end
