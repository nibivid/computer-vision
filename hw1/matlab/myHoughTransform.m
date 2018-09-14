function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)
%Your implementation here
%Im - grayscale image - 
%threshold - prevents low gradient magnitude points from being included
%rhoRes - resolution of rhos - scalar
%thetaRes - resolution of theta - scalar
    
    % threshold
    [rows, cols] = find(Im>threshold);
    voteNum = size(rows,1);
    
    % number of bins
    thetaNum = floor(2*pi/thetaRes)+1;  
    thetaScale = 0 : thetaRes : 2*pi;
    
    [height, width] = size(Im);
    rhoMax = sqrt(height*height+width*width) + rhoRes;
    rhoNum = floor(rhoMax/rhoRes);
    rhoScale = 0 : rhoRes : rhoMax;
    
    % accumulate
    H = zeros(rhoNum, thetaNum);
    for i = 1:voteNum
        x = cols(i);
        y = rows(i);
        for t = 1:thetaNum
            theta = (t-1) * thetaRes;
            rho = x*cos(theta) + y*sin(theta);
            if rho < 0
                continue;
            end
            rho_idx = floor(rho/rhoRes) + 1;
            H(rho_idx,t) = H(rho_idx,t) + 1;
        end
    end
%     figure(1)
%     H_img = uint8(256/max(max(H))*H);
%     imshow(H_img)
%     Im(Im<threshold) = 0;
%     [H_gt, T_gt, R_gt] = hough(Im,'RhoResolution',rhoRes,'Theta',-90:thetaRes*180/pi:89);
%     figure(2)
%     H_gt_img = uint8(256/max(max(H_gt))*H_gt);
%     imshow(H_gt_img)
end
        
        
