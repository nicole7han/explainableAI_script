% create stimuli given parameters

function [stimuli, radius, aspect_ratio, angle] = create_stimuli(s_r, ar_r,or_r, imageSizeX, imageSizeY)

%      s: size
%     s_r: size range
%     ar: aspect ratio
%     ar_r: aspect ratio range
%     or: orientation
%     or_r: orientation range

    [columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
    centerX = (imageSizeX+1)/2;
    centerY = (imageSizeY+1)/2;

    radius = randi( s_r); % radius
    aspect_ratio = ar_r(1) + (ar_r(2)-ar_r(1))*rand(1); % aspect ratio
    b = radius;
    a = radius*aspect_ratio;
    theta = randi( or_r)*pi/180; % orientation
    angle = round(180-theta*180/pi);

    % approach1
    img = double(( (columnsInImage - centerX)*cos(theta)+(rowsInImage - centerY)*sin(theta) ).^2/a^2 + ...
          ( (columnsInImage - centerX)*sin(theta)-(rowsInImage - centerY)*cos(theta) ).^2/b^2 ...
             <= 1)*150/255;
    img(img==0)=.5;

    sigma=.08;
    stimuli = img + sigma*randn(size(img));


    % % approach2
    % img_n2 = imnoise(img, 'gaussian', 0, sigma^2);
    % imshow(img_n2);
    % img_n2 = imnoise(img/255,'gaussian',0,(sigma/255)^2);
    % imshow(uint8(img_n2*255));

    % % approach3
    % img_n3 = awgn(img,var(img(:))/sigma);
    % imshow(uint8(img_n3));

end