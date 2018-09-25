function [points] = getRandomPoints(I, alpha)
% Generates random points in the image
% Input:
%   I:                      grayscale image
%   alpha:                  random points
% Output:
%   points:                    point locations
%
	% -----fill in your implementation here --------
    h = size(I, 1);
    w = size(I, 2);
    ph = randi(h, alpha, 1);
    pw = randi(w, alpha, 1);
    
    points = zeros(alpha, 1);
    points(:, 1) = ph;
    points(:, 2) = pw;
    
%     imshow(I);
%     hold on
%     plot(pw, ph, 'r.');

    % ------------------------------------------

end

