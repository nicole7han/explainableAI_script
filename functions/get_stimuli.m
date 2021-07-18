function stimuli = get_stimuli(s, ar, theta, imageSizeX, imageSizeY)
    
    [columnsInImage rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);
    centerX = (imageSizeX+1)/2;
    centerY = (imageSizeY+1)/2;
    
    b = s;
    a = b * ar;
    angle = round(180-theta*180/pi);
    
    img = double(( (columnsInImage - centerX)*cos(theta)+(rowsInImage - centerY)*sin(theta) ).^2/a^2 + ...
          ( (columnsInImage - centerX)*sin(theta)-(rowsInImage - centerY)*cos(theta) ).^2/b^2 ...
             <= 1) ...
             *140/255;
    img(img==0)=.5;

    sigma=.1;
    stimuli = img + sigma*randn(size(img));
%     stimuli = imadjust(stimuli,[0 1],[0.3 0.7]); 
end