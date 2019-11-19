% filedir_BF = 'C:\Users\diederichbenedict\Desktop\15 calibration fibre 38\BF\IMG_20170621_141309.jpg'
% out = readtimeseries(filedir_BF)
%
%
%
%
% % Get list of all BMP files in this directory
% % DIR returns as a structure array.  You will need to use () and . to get
% % the file namegs.
%
%
% imagefiles = dir('*.bmp');
% nfiles = length(imagefiles);    % Number of files found
% for ii=1:nfiles
%    currentfilename = imagefiles(ii).name;
%    currentimage = imread(currentfilename);
%    images{ii} = currentimage;
% end

close all




% Iav = dip_image(1.*imread('IMG_20170621_141309.dng'));
% Bav = dip_image(1.*imread('IMG_20170621_141420.jpg'));
%

Iav_file_name='IMG_20170621_141309.dng';
Iav = dip_image((getRawImg(Iav_file_name)));
Bav = dip_image(ones(size(Iav)).*mean(Iav))

% remove some dust
Iav_filtered = 0*Iav;
for(i=0:2)
    Iav_filtered(:,:,i) = gaussf(Iav(:,:,i), 1);
end
Iav = Iav_filtered;


if(~exist('ROIPosition'))
    %% create a figure handle and select the coordinates
    fh=dipshow(Iav(:,:,1)); % find edges of CC signal
    diptruesize(fh, 50);
    %fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
    fprintf('Please select the center of the ROI you want to process, then click on the outer rim of the ROI.');
    ROIPosition = dipgetcoords(fh,2);
    ROICentre = ROIPosition(1,:);
    ROIOuter = ROIPosition(2,:);
    ROIDiff = abs(ROICentre-ROIOuter);
    ROIDiff = [max(ROIDiff) max(ROIDiff)];
end

Iav = extract(Iav, 2*ROIDiff, ROICentre);
Bav = extract(Bav, 2*ROIDiff, ROICentre);

% get RGB histogram
input = double(Iav);
hist(reshape(input,[],3),1:max(input(:)));
colormap([1 0 0; 0 1 0; 0 0 1])

%% https://de.mathworks.com/help/images/reduce-the-number-of-colors-in-an-image.html
% make sure double is normalized between 0..1
n_colours = 1024;
[X_no_dither,map]= rgb2ind(dip_array(Iav)./max(max(max(dip_array(Iav)))),n_colours ,'nodither');
dip_image(X_no_dither)
histogram(X_no_dither)


figure, imshow(X_no_dither,map);
rgbImage = ind2rgb(X_no_dither, map);
Iav = Iav;dip_image(rgbImage)

% I_subtractR = double(1.*Iav{1})./double(1.*Bav{1});
% I_subtractR(isnan(I_subtractR))=1;
% I_subtractR(isinf(I_subtractR))=1;
%
% cat(3, Iav{1}, Bav{1},dip_image(I_subtractR))

mysize=size(squeeze(Iav(:,:,1)));

I_subtract = Iav;%./(Bav);
I_subtract = extract(I_subtract, [mysize(1) mysize(1)], []);


%col=COLORSPACE(Iav);
%I_subtract=JOINCHANNELS(col,I_subtractR,I_subtractG,I_subtractB);



if(~exist('FiberPosition'))
    %% create a figure handle and select the coordinates
    fh=dipshow(I_subtract(:,:,1)); % find edges of CC signal
    diptruesize(fh, 200);
    %fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
    fprintf('Please select 4 coordinates which describe the position of the fiber');
    fprintf('Order: Upper-Right Corner, Lower-Right Corner, Lower-Left Corner, Upper-Left Corner')
    FiberPosition = dipgetcoords(fh,4);
    fprintf('Thank you :-)')
end
%% get angle of the fiber laying in the sample
% in dip-coordinates the first is X, wheras in Matlab it's y-coordinate
alpha_left = atan((FiberPosition(1,2)-FiberPosition(4,2))/(FiberPosition(1,1)-FiberPosition(4,1)));
alpha_right = atan((FiberPosition(2,2)-FiberPosition(3,2))/(FiberPosition(2,1)-FiberPosition(3,1)));
alpha_avg = mean([alpha_left, alpha_right]);

center_left = FiberPosition(1,:)+(FiberPosition(4,:)-FiberPosition(1,:))/2;
center_right = FiberPosition(2,:)+(FiberPosition(3,:)-FiberPosition(2,:))/2;

% calculate the diameter of the fiber (Pythagoras)
dist_left_right = sqrt((center_left(1)-center_right(1)).^2+(center_left(2)-center_right(2)).^2);

% calculate the central position of the fiber
center_fiber = center_left+(center_right-center_left)/2;

% debug: Show line
length_line = 400;

x1 = -sin(alpha_avg+pi/2)*length_line+center_fiber(1);
y1 = +cos(alpha_avg+pi/2)*length_line+center_fiber(2);

x2 = sin(alpha_avg+pi/2)*length_line+center_fiber(1);
y2 = -cos(alpha_avg+pi/2)*length_line+center_fiber(2);
line([x1,x2],[y1,y2])


%% draw a fiber
center_fiber_xy = round(center_fiber-mysize/2);
xy_grid = -cos(alpha_avg)*(yy(mysize)-center_fiber_xy(2))+sin(alpha_avg)*(xx(mysize)-center_fiber_xy(1));
%xy_grid = circshift((xy_grid), round(center_fiber-size(I_subtractR)/2))
fiber_shape = real(sqrt((dist_left_right/2).^2-(xy_grid).^2));

cat(3, I_subtract, fiber_shape)

%% select ROI

I_mask = drawpolygon(newim(size(squeeze(I_subtract(:,:,1)))),FiberPosition,1,'closed');
I_mask = dip_image(I_mask,'bin');
I_mask = ~bpropagation(I_mask&0,~I_mask,0,1,1);

I_subtract= I_subtract*I_mask;
fake_fiber= fiber_shape*I_mask;
X_no_dither= X_no_dither*I_mask;

cat(3, I_subtract, fake_fiber);


if(0)
    % rotate object
    theta = alpha_avg/pi*180;
    tform = affine2d([cosd(theta) -sind(theta) 0; sind(theta) cosd(theta) 0; 0 0 1]);
    fake_fiber = dip_image(imwarp(double(fake_fiber),tform));
    I_subtract = dip_image(imwarp(double(I_subtract),tform));
end

if(0)
    % further reduce ROI
    %% create a figure handle and select the coordinates
    fh=dipshow(I_subtract(:,:,1)); % find edges of CC signal
    diptruesize(fh, 100);
    %fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
    fprintf('Please select the center of the ROI you want to process, then click on the right rim of the ROI.');
    ROIPosition = dipgetcoords(fh,2);
    ROICentre = ROIPosition(1,:);
    ROIOuter = ROIPosition(2,:);
    ROIDiff = abs(ROICentre-ROIOuter);
    ROIDiff = [max(ROIDiff) max(ROIDiff)];
    
    Iav = extract(Iav, 2*ROIDiff, ROICentre);
    Bav = extract(Bav, 2*ROIDiff, ROICentre);
    
end


%% fit data

for opdfact = 3;%linspace(2,4,20)
    lambdaG = 550;
    opdmax= opdfact * lambdaG
    obj=fake_fiber/max(fake_fiber) * opdmax;
    
    colMix = SimCol(obj);
    %showCol(colMix)
    
    
    
    
    %cat(3, I_subtract(:,:,0), colMix(:,:,0), I_subtract(:,:,1), colMix(:,:,1), I_subtract(:,:,2), colMix(:,:,2))
    
    % apply data
    I_mes = I_subtract;%cat(3, I_subtract{1}, I_subtract{2}, I_subtract{3})
    
    % export for NN RGB
    if(false)
        obj_NN = double(reshape(obj, [1, numel(double(obj))]));
        I_mes_R_NN = double(reshape(I_subtract(:,:,0), [1, numel(double(obj))]));
        I_mes_G_NN = double(reshape(I_subtract(:,:,1), [1, numel(double(obj))]));
        I_mes_B_NN = double(reshape(I_subtract(:,:,2), [1, numel(double(obj))]));
        I_mes_NN = cat(2, I_mes_R_NN, I_mes_G_NN, I_mes_B_NN);
        save('JAMIN_LBF_NN', 'obj_NN', 'I_mes_NN', '-v7.3')
    end
    
    
    % export for NN rgb2ind
    if(true)
        cat(3, obj, dip_image(X_no_dither))
        obj_NN = double(reshape(obj, [1, numel(double(obj))]));
        orig_rgb2ind = double(reshape(X_no_dither, [1, numel(double(obj))]));
        save('JAMIN_LBF_NN_rgb2ind_v73', 'obj_NN', 'orig_rgb2ind', '-v7.3')
    end
    
    
    
    save('JAMIN_LBF_NN_2D', 'obj', 'I_mes', '-v7.3')
    
    
    [OPDaxis,R,G,B]=getCalibration(obj,I_mes,n_colours/8);
    
    
    
    if(true)
        figure
        plot(R)
        hold on
        plot(G), plot(B)
        hold off
        legend('R', 'G', 'B')
        xlabel('OPD')
        ylabel('Intensity')
        
        plot3(R, G, B)
        xlabel 'R'
        ylabel 'G'
        zlabel 'B'
        
        
        if(1)
            [OPDaxis_,R_,G_,B_]=getCalibration(obj,colMix,256);
            
            plot3(R_, G_, B_)
            xlabel 'R'
            ylabel 'G'
            zlabel 'B'
            OPDMap_ = findOPD(colMix,R_,G_,B_,OPDaxis_,[],[0 0 0])
            
            plot3(R_, G_, B_)
            xlabel 'R'
            ylabel 'G'
            zlabel 'B'
            
        end
        
        
    end
    %OPDMap = findOPD(I_mes,R,G,B,OPDaxis,[],[1 1 0])
    OPDMap = findOPD(I_mes,R,G,B,OPDaxis,[],[0 0 0])
    cat(3, OPDMap, I_mes)
end




return

%% apply to real data
Iob_file_name = 'IMG_20170621_160615.dng';
Iob = dip_image((getRawImg(Iob_file_name)))
Bob = dip_image(ones(size(Iob)).*mean(Iob));%dip_image(1.*imread('IMG_20170621_141420.jpg'));


% create a figure handle and select the coordinates
fh=dipshow(Iob(:,:,1)); % find edges of CC signal
diptruesize(fh, 50);
%fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
fprintf('Please select centre of ROI');
ROIPosition = dipgetcoords(fh,2);
ROICentre = ROIPosition(1,:);
ROIOuter = ROIPosition(2,:);
ROIDiff = abs(ROICentre-ROIOuter);
ROIDiff = [max(ROIDiff) max(ROIDiff)];


Iob = extract(Iob, 2*ROIDiff, ROICentre);
Bob = extract(Bob, 2*ROIDiff, ROICentre);

I_ob_subtract = Iob;%./(Bob);

OPDMap_ob = findOPD(I_ob_subtract ,R,G,B,OPDaxis,[],[1 1 0])
cat(3, OPDMap_ob, Iob(:,:,1))
