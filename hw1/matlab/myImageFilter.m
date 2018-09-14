function [img1] = myImageFilter(img0, h)
    % built-in function
    img1_gt = conv2(img0, h, 'same');
    
    [ks_h, ks_w] = size(h);
    ps_h = floor((ks_h - 1) / 2);
    ps_w = floor((ks_w - 1) / 2);
    [height, width] = size(img0);
    
    % opposite the kernel
    h = rot90(h, 2);
    
    % pad with nearest value
    img = zeros(2*ps_h+height, 2*ps_w+width);
    img(ps_h+1:ps_h+height, ps_w+1:ps_w+width) = img0;
    pad_up = repmat(img0(1,:),[ps_h, 1]);
    img(1:ps_h,ps_w+1:ps_w+width) = pad_up;
    pad_down = repmat(img0(height,:),[ps_h, 1]);
    img(height+ps_h+1:height+2*ps_h,ps_w+1:ps_w+width) = pad_down;
    pad_left = repmat(img0(:,1),[1, ps_w]);
    img(ps_h+1:ps_h+height,1:ps_w) = pad_left;
    pad_right = repmat(img0(:,width),[1, ps_w]);
    img(ps_h+1:ps_h+height,width+ps_w+1:width+2*ps_w) = pad_right;
    pad_lu = repmat(img0(1,1), [ps_h, ps_w]);
    img(1:ps_h,1:ps_w) = pad_lu;
    pad_ld = repmat(img0(height,1), [ps_h, ps_w]);
    img(height+ps_h+1:height+2*ps_h,1:ps_w) = pad_ld;
    pad_ru = repmat(img0(1,width), [ps_h, ps_w]);
    img(1:ps_h,width+ps_w+1:width+2*ps_w) = pad_ru;
    pad_rd = repmat(img0(height,width), [ps_h, ps_w]);
    img(height+ps_h+1:height+2*ps_h,width+ps_w+1:width+2*ps_w) = pad_rd;
    
    % convolve
    img_cols = im2col(img, [ks_h, ks_w], 'sliding');
    h_cols = h(:);
    img_conv = h_cols' * img_cols;
    img1 = reshape(img_conv,[height, width]);
    
    diff = img1 - img1_gt;
    figure(1)
    imshow(uint8(img1));
end