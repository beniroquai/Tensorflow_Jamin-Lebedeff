close all

% load darkframe
I_dark = dip_image(getRawImg('/Users/Bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/MATLAB/DARKFRAME.dng'));
I_dark = mean(I_dark);
% load file
imfolder = '/Volumes/Macintosh HD/Users/Bene/Pictures/JAMIN LEBEDEFF/17_fiber_calibration_series/';
imfolder = '/Volumes/Macintosh HD/Users/Bene/Pictures/JAMIN LEBEDEFF/19_Calibration_Fiber_New/Calibration_Fiber/'

% look for all images in folder
imagefilesDNG = dir(strcat(imfolder, '*.jpg'));
nFiles = length(imagefilesDNG);    % Number of files found

if(~exist('Iav'))
    Iav=0;
    for(ii = 1:nFiles)
        
        %Iav = Iav + dip_image(getRawImg([imfolder imagefilesDNG(ii).name;]));
        Iav = Iav + dip_image(imrotate(imread([imfolder imagefilesDNG(ii).name]), 90));
    end
    Iav = Iav/nFiles-I_dark;
end

% select a sub-roi
if(~exist('ROIPosition'))
    %% create a figure handle and select the coordinates´+´0ßoq1    >    
    fh=dipshow(Iav(:,:,1)); % find edges of CC signal
    diptruesize(fh, 30);
    %fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
    fprintf('Please select the center of the ROI you want to process, then click on the outer rim of the ROI.');
    ROIPosition = dipgetcoords(fh,2);
    ROICentre = ROIPosition(1,:);
    ROIOuter = ROIPosition(2,:);
    ROIDiff = abs(ROICentre-ROIOuter);
    ROIDiff = [max(ROIDiff) max(ROIDiff)];
end

% try to normalize blue colour channel
Iav = extract(Iav, 2*ROIDiff, ROICentre);

Bav = dip_image(ones(size(Iav)).*mean(Iav));
Bav = extract(Bav, 2*ROIDiff, ROICentre);

mysize=size(squeeze(Iav(:,:,1)));

I_subtract = Iav;%./(Bav);
I_subtract = extract(I_subtract, [mysize(1) mysize(1)], []);


%% Generate a fake fiber object
if(~exist('FiberPosition'))
    %% create a figure handle and select the coordinates
    fh=dipshow(I_subtract(:,:,1)); % find edges of CC signal
    diptruesize(fh, 100);
    %fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
    fprintf('Please select 4 coordinates which describe the position of the fiber');
    fprintf('Order: Upper-Right Corner, Lower-Right Corner, Lower-Left Corner, Upper-Left Corner')
    FiberPosition = dipgetcoords(fh,4);
    fprintf('Thank you :-)')
end

[fiber_shape, theta] = getFakeFiber(FiberPosition, mysize);
cat(3, I_subtract, fiber_shape)

%% select ROI
I_mask = drawpolygon(newim(size(squeeze(I_subtract(:,:,1)))),FiberPosition,1,'closed');
I_mask = dip_image(I_mask,'bin');
I_mask = ~bpropagation(I_mask&0,~I_mask,0,1,1);





%% fit data


%% Experiment with the separation of the colourchannels
Imes = I_mask*I_subtract;
Imesbluemean = mean(Imes);
Imes(Imes>210) = Imesbluemean ;
Imesblue = Imes(:,:,2);
Imesbluemin = min(Imesblue(I_mask));
Imesblue = (Imesblue-Imesbluemin);
Imes(:,:,2) = Imesblue*5;
Imes(Imes>250) = Imesbluemean ;

% Blur has the major effect of separating the colour channels!
if(false)
    Imes(:,:,0) = gaussf(Imes(:,:,0), 1);
    Imes(:,:,1) = gaussf(Imes(:,:,1), 1);
    Imes(:,:,2) = gaussf(Imes(:,:,2), 1);
else
    len = 500;
    theta_ang = -theta/pi*180;
    PSF = fspecial('motion', len, theta_ang);
    PSF = (PSF>0)*mean(mean(PSF));
    
    Imes(:,:,0) = dip_image(imfilter(double(Imes(:,:,0)), PSF, 'circular'));
    Imes(:,:,1) = dip_image(imfilter(double(Imes(:,:,1)), PSF, 'circular'));
    Imes(:,:,2) = dip_image(imfilter(double(Imes(:,:,2)), PSF, 'circular'));
    
end
Imes = Imes*I_mask



%% get histogram
try
input = double(Imes);
hist(reshape(input,[],3),1:max(input(:)));
colormap([1 0 0; 0 1 0; 0 0 1])
catch
end

Imes= Imes*I_mask;
fake_fiber= fiber_shape*I_mask;

opdfact = 3;%linspace(2,4,20)
lambdaG = 550;
opdmax= opdfact * lambdaG
obj=fake_fiber/max(fake_fiber) * opdmax;

colMix = SimCol(obj);
%showCol(colMix)

%% Try out max/sum projection
%obj = sum(obj, [], 1)
%obj = repmat(obj, [size(obj,2), 1])
%Imes = repmat(Imes, [size(Imes,2), 1,1])
%Imes = dip_image(imrotate(double(Imes), -theta_ang));
%Imes = sum(Imes, [], 1)


[OPDaxis,R,G,B]=getCalibration(obj,Imes,256);
OPDMap = findOPD(Imes,R,G,B,OPDaxis,[],[0 0 0])

figure
plot3(R, G, B)
xlabel 'R'
ylabel 'G'
zlabel 'B'

if(true)
    
    figure
    plot(R)
    hold on
    plot(G), plot(B)
    hold off
    legend('R', 'G', 'B')
    xlabel('OPD')
    ylabel('Intensity')
end
OPDMap = findOPD(Imes,R,G,B,OPDaxis,[],[0 0 0])


%% fit spline curve to data and watch the effect
xyz = cat(2, R', G', B');
xyzfitpt = xyz(1:9:end, :);
xyzfit = cscvn(xyzfitpt');
t_range = linspace(-2*min(min(xyz)), 5*max(max(xyz)), numel(R));
xyzfiteval = fnval(xyzfit, t_range);
plot3(xyzfiteval(1, :), xyzfiteval(2, :), xyzfiteval(3, :))

R_fit = xyzfiteval(1, :);
G_fit = xyzfiteval(2, :);
B_fit = xyzfiteval(3, :);

OPDMap = findOPD(Imes,R_fit,G_fit,B_fit,OPDaxis,[],[0 0 0])

%%
figure
subplot(121)
plot3(R_fit, G_fit, B_fit, 'b')
xlabel 'R'
ylabel 'G'
zlabel 'B'

hold on
plot3(R, G, B, 'g')
hold off
legend ('RGB fit', 'RGB raw')

subplot(122)
fnplt(cscvn(xyzfitpt'), 'r')

%% apply to real data
Iob_file_name = 'IMG_20170621_160615.dng';
Iob_file_name = '/Volumes/Macintosh HD/Users/Bene/Pictures/JAMIN LEBEDEFF/17 images 2/IMG_20170721_153911.dng'
Iob = dip_image((getRawImg(Iob_file_name))) - I_dark;
Bob = dip_image(ones(size(Iob)).*mean(Iob));%dip_image(1.*imread('IMG_20170621_141420.jpg'));




% create a figure handle and select the coordinates
fh=dipshow(dip_image(rgb2gray(double(Iob)/255))); % find edges of CC signal
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


OPDMap_ob = findOPD(I_ob_subtract ,R,G,B,OPDaxis,[],[0 0 0])
cat(3, OPDMap_ob, Iob(:,:,1))

% get RGB histogram
Iob(Iob>210) = Imesbluemean;
Iobblue = Iob(:,:,2);
Iobblue = (Iobblue-Imesbluemin);
Iob(:,:,2) = Iobblue*5;
Iob(Iob>250) = Imesbluemean;


    Iob(:,:,0) = gaussf(Iob(:,:,0), 5);
    Iob(:,:,1) = gaussf(Iob(:,:,1), 5);
    Iob(:,:,2) = gaussf(Iob(:,:,2), 5);

OPDMap_ob = findOPD(Iob ,R,G,B,OPDaxis,[],[0 0 0])
cat(3, OPDMap_ob, Iob(:,:,1))




input = double(Iob);
hist(reshape(input,[],3),1:max(input(:)));
colormap([1 0 0; 0 1 0; 0 0 1])


if(0)
    %%
    colMix = colMix*reshape([1 1 2], [1 1 3]);
    colMix(:,:,2) = (colMix(:,:,2))^2;
    [OPDaxis_,R_,G_,B_]=getCalibration(obj,colMix,256);
    
    figure
    plot3(R_, G_, B_)
    xlabel 'R'
    ylabel 'G'
    zlabel 'B'
    OPDMap_ = findOPD(colMix,R_,G_,B_,OPDaxis_,[],[0 0 0])
    
    
    figure
    plot(R_)
    hold on
    plot(G_), plot(B_)
    hold off
    legend('R', 'G', 'B')
    xlabel('OPD')
    ylabel('Intensity')
end
%%

xyz = cat(2, R', G', B');
xyzfitpt = xyz(1:9:end, :);
xyzfit = cscvn(xyzfitpt');
t_range = linspace(-min(min(xyz)), 5*max(max(xyz)), 100);
xyzfiteval = fnval(xyzfit, t_range);


R_ = xyzfiteval(1, :);
G_ = xyzfiteval(2, :);
B_ = xyzfiteval(3, :);


figure
subplot(121)
plot3(R_, G_, B_, 'b')
xlabel 'R'
ylabel 'G'
zlabel 'B'

subplot(122)
fnplt(cscvn(xyzfitpt'), 'r')
