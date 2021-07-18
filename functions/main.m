
%% create target
sig_s = 30;
sig_s_r = [sig_s-1,sig_s+1];
sig_ar = 1.8;
sig_ar_r = [sig_ar-.1,sig_ar+.1];
sig_or = 45;
sig_or_r = [180-sig_or-5,180-sig_or+5]; %range(sig_or-5,sig_or+5)

imageSizeX = 255*3;
imageSizeY = 255*3;

for i = 1:10
    [stimuli,radius,aspect_ratio,orien] = create_stimuli(sig_s_r, sig_ar_r, sig_or_r, imageSizeX, imageSizeY);
    imshow(stimuli);
    saveas(gcf, sprintf('Stimuli/target_r_%d_ar_%.2f_or_%d.jpg',radius,round(aspect_ratio,1),orien));
    close;
end


%% create distractor
dis_s = 20;
dis_s_r = [dis_s-1,dis_s+1];
dis_ar = 1.6;
dis_ar_r = [dis_ar-.1,dis_ar+.1];
dis_or = 30;
dis_or_r = [180-dis_or-5,180-dis_or+5]; %range(sig_or-5,sig_or+5)


for i = 1:10
    [stimuli,radius,aspect_ratio,orien] = create_stimuli(dis_s_r,dis_ar_r,dis_or_r,imageSizeX,imageSizeY);
    imshow(stimuli);
    saveas(gcf, sprintf('Stimuli/distractor_r_%d_ar_%.2f_or_%d.jpg',radius,round(aspect_ratio,1),orien));
    close;
end


%% create targets and distractors with sphere

imageSizeX = 255;
imageSizeY = 255;

N = 20; % sqrt(#stimuli) in each category
num_stimuli = N^2;

[X,Y,Z] = sphere(N-1); % create N points on a unit sphere;
x = [X(:); 2.5*X(:)];
y = [Y(:); 1.5*Y(:)];
z = [Z(:); 2.5*Z(:)];

sizes = repmat([20,20],numel(X),1);
colors = repmat([1,2],numel(X),1);

targets = [x(1:num_stimuli)+20, y(1:num_stimuli)+1.5, z(1:num_stimuli)+45]; 
distractors = [x(num_stimuli+1:end)+20, y(num_stimuli+1:end)+1.5, z(num_stimuli+1:end)+45]; 
x = [targets(:,1); distractors(:,1)];
y = [targets(:,2); distractors(:,2)];
z = [targets(:,3); distractors(:,3)];

figure
scatter3(x,y,z,sizes(:),colors(:),'filled');

target_range = [min(targets); max(targets)];
distractor_range = [min(distractors); max(distractors)];

for i = 1:10
    % target
    Xcell=num2cell([targets(i,1),targets(i,2),targets(i,3)]);
    [s, ar, ang]=deal(Xcell{:});
    s = round(s);
    theta = (180-ang)*pi/180;
    stimuli = get_stimuli(s, ar, theta, imageSizeX, imageSizeY);
    imshow(stimuli);
    saveas(gcf, sprintf('Stimuli/target_s_%d_ar_%.2f_or_%.2f.jpg',s,round(ar,1),ang));
    close;
    
    % distractor
    Xcell=num2cell([distractors(i,1),distractors(i,2),distractors(i,3)]);
    [s, ar, ang]=deal(Xcell{:});
    s = round(s);
    theta = (180-ang)*pi/180;
    stimuli = get_stimuli(s, ar, theta, imageSizeX, imageSizeY);
    imshow(stimuli);
    saveas(gcf, sprintf('Stimuli/distractor_s_%d_ar_%.2f_or_%.2f.jpg',s,round(ar,1),ang));
    close;
end

% 
% %% visualize target & distractor
% %sample target
% targets = []; distractors = [];
% for i = 1:20
%     [tar,radius,aspect_ratio,orien] = create_stimuli(sig_s_r, sig_ar_r, sig_or_r, imageSizeX, imageSizeY);
%     targets = [targets; radius, aspect_ratio, orien];
%     [dis,radius,aspect_ratio,orien] = create_stimuli(dis_s_r,dis_ar_r,dis_or_r,imageSizeX,imageSizeY);;
%     distractors = [distractors; radius, aspect_ratio, orien];
% end
% 
% sizes = repmat([35,35],size(targets,1),1);
% colors = repmat([1,2],size(targets,1),1);
% x = [targets(:,1); distractors(:,1)];
% y = [targets(:,2); distractors(:,2)];
% z = [targets(:,3); distractors(:,3)];
% figure
% scatter3(x,y,z,sizes(:),colors(:),'filled');
% view(40,35)
