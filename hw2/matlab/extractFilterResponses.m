function [filterResponses] = extractFilterResponses(I, filterBank)
% CV Fall 2018 - Provided Code
% Extract the filter responses given the image and filter bank
% Pleae make sure the output format is unchanged.
% Inputs:
%   I:                  a 3-channel RGB image with width W and height H
%   filterBank:         a cell array of N filters
% Outputs:
%   filterResponses:    a HxWx3N matrix of filter responses


    %Convert input Image to Lab
    doubleI = double(I);
    if length(size(doubleI)) == 2
        tmp = doubleI;
        doubleI(:,:,1) = tmp;
        doubleI(:,:,2) = tmp;
        doubleI(:,:,3) = tmp;
    end
    [L,a,b] = RGB2Lab(doubleI(:,:,1), doubleI(:,:,2), doubleI(:,:,3));
    h = size(I,1);
    w = size(I,2);

   
    % -----fill in your implementation here --------
    filterResponses = zeros(h,w,3*numel(filterBank));
    for idx = 1 : numel(filterBank)
        filter = filterBank{idx,1};
%         imshow(L, []);
%         imshow(a, []);
%         imshow(b, []);
        L_res = imfilter(L, filter);
        a_res = imfilter(a, filter);
        b_res = imfilter(b, filter);
%         imshow(L_res, []);
%         imshow(a_res, []);
%         imshow(b_res, []);
        filterResponses(:,:,3*(idx-1)+1) = L_res;
        filterResponses(:,:,3*(idx-1)+2) = a_res;
        filterResponses(:,:,3*idx) = b_res;
    end
    
    % ------------------------------------------
end
