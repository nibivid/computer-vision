function [rhos, thetas] = myHoughLines(H, nLines)
%Your implemention here
    [height, width] = size(H);
    
    % NMS
    H_nms = H;
    for h = 2:height-1
        for w = 2:width-1       
            if H(h,w)<H(h-1,w-1)||H(h,w)<H(h-1,w)||H(h,w)<H(h-1,w+1)||...
                    H(h,w)<H(h,w-1)||H(h,w)<H(h,w+1)||H(h,w)<H(h+1,w-1)...
                    ||H(h,w)<H(h+1,w)||H(h,w)<H(h+1,w+1)
                H_nms(h,w) = 0;
            end
        end
    end
    H = H_nms;
%     figure(3)
%     imshow(H)
    
    % find peak
    rhos = zeros(nLines, 1);
    thetas = zeros(nLines, 1);
    H_flat = H(:);
    [H_sort, idx] = sort(H(:), 'descend');
    for i = 1:nLines
        idx_flat = idx(i);
        idx_theta = ceil(idx_flat/height);
        idx_rho = idx_flat-(idx_theta-1)*height;
        rhos(i) = idx_rho;
        thetas(i) = idx_theta;
    end
    
end
        