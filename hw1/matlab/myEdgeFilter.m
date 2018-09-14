function [Im Io Ix Iy] = myEdgeFilter(img, sigma)
%Your implemention
    % guassian blurring
    hsize = 2 * ceil(3*sigma) + 1;
    h_gaussian = fspecial('gaussian',hsize,sigma);
    img_blur = myImageFilter(img, h_gaussian);
    
    % image gradient in x and y direction
    h_sobel_x = fspecial('sobel');
    Ix = myImageFilter(img_blur, h_sobel_x);
    h_sobel_y = h_sobel_x';
    Iy = myImageFilter(img_blur, h_sobel_y);
    
    % gradient magnitude and direction
    Im = sqrt(Ix.*Ix + Iy.*Iy);
    Io = atan(Iy./Ix);
    
    % NMS to make Im thinner
    [height, width] = size(img);
%     figure(1)
%     imshow(img);
%     figure(2)
%     imshow(uint8(Im));
    
    Im_nms = Im;
    for i = 2 : height-1
        for j = 2 : width-1
            theta = Io(i,j);    %(-pi/2,pi/2)
            if theta<=-3*pi/8 || theta>=3*pi/8    %(-pi/2,-3pi/8)&(3pi/8,pi/2)
                if Im(i,j-1)>Im(i,j) || Im(i,j+1)>Im(i,j)
                    Im_nms(i,j)=0;
                end
            end
            if theta<=-pi/8 && theta>-3*pi/8    %(-3pi/8,-pi/8)
                if Im(i-1,j+1)>Im(i,j) || Im(i+1,j-1)>Im(i,j)
                    Im_nms(i,j)=0;
                end
            end
            if theta<=pi/8 && theta>-pi/8    %(-pi/8,pi/8)
                if Im(i-1,j)>Im(i,j) || Im(i+1,j)>Im(i,j)
                    Im_nms(i,j)=0;
                end
            end
            if theta<3*pi/8 && theta>pi/8    %(pi/8,3pi/8)
                if Im(i-1,j-1)>Im(i,j) || Im(i+1,j+1)>Im(i,j)
                    Im_nms(i,j)=0;
                end
            end
        end
    end
    Im = Im_nms;
    
%     figure(3)
%     imshow(uint8(Im));
end
    
                
        
        
