function [points] = getHarrisPoints(I, alpha, k)
% Finds the corner points in an image using the Harris Corner detection algorithm
% Input:
%   I:                      grayscale image
%   alpha:                  number of points
%   k:                      Harris parameter
% Output:
%   points:                    point locations
%
    % -----fill in your implementation here --------
    % uint8 to double
    I = im2double(I);
    if ndims(I) == 3
        I_grayscale = rgb2gray(I);
    else
        I_grayscale = I;
    end
    
    % sobel x y to get Ix Iy
    h_sobel_x = fspecial('sobel');
    h_sobel_y = h_sobel_x';
    Ix = imfilter(I_grayscale, h_sobel_x);
    Iy = imfilter(I_grayscale, h_sobel_y);
    Ix = Ix - mean(mean(Ix));
    Iy = Iy - mean(mean(Iy));
    
    % compute H
    h_average = fspecial('average', 5);
    IxIx = imfilter(Ix.*Ix, h_average);
    IyIy = imfilter(Iy.*Iy, h_average);
    IxIy = imfilter(Ix.*Iy, h_average);
    
    % compute eigenvalue and sort alpha corners
    R = IxIx.*IyIy - IxIy.*IxIy - k * (IxIx + IyIy).*(IxIx + IyIy);
    [R_sort, select_index] = sort(R(:), 'descend');
    select_index = select_index(1 : alpha);
    
    % find harris points
    [h, w] = size(I_grayscale);
    pw = floor(select_index / h) + 1;
    ph = select_index - h * (pw-1);
    pw(ph<=0) = pw(ph<=0) - 1;
    ph(ph<=0) = h;
    
    points = zeros(alpha, 1);
    points(:, 1) = ph;
    points(:, 2) = pw;
    
    % plot
%     imshow(I);
%     hold on
%     plot(pw, ph, 'r.');
    

    % ------------------------------------------

end
