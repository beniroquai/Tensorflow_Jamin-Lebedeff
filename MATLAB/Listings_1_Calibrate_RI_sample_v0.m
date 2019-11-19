%% This file tries to recover the OPD of a sample with known reference
% It takes two cellphone images, where one is a known fiber-like object
% Authors: B. Diederich, B. Marsikova, R. Heintzmann
close all

%% Add some parameters here
is_dust_particle_remove = false; % do you want to remove dust particles in the ref. image?
is_regularization = false; % eventually regularize the input image (blur it)
is_splinefit = false; % seperate the color channels - only possible with fitting toolbox
n_opdsteps = 50;

% add the library
addpath('./src')

% determine the path for the images 
imfolder = './data/EXP1/'
f_ref_img = 'IMG_20191118_155446.jpg';
f_ref_img_raw = 'IMG_20191118_154528.dng';
f_sample_im = 'IMG_20191118_160025.jpg'; 



% load the imagefiles
%I_ref_raw = dip_image(flip(imread([imfolder, f_ref_img])));
I_ref_raw = dngRead([imfolder, f_ref_img_raw]);
flip(extracBayerChannel(I_ref_raw))

%% extract the Ref.-ROI
I_ref = JLextractROI(I_ref_raw);
mysize=size(squeeze(I_ref(:,:,1)));

%% remove dust particles
if(is_dust_particle_remove)
    I_ref = JLremoveDust(I_ref, 0, 51);
end



%% fit the artificial fiber in the experimental dataset
[fiber_shape, theta, mask] = JLFitFibre(I_ref);

% Visualize the groundtruth and its measurement
cat(3, I_ref , fiber_shape)

%% Eventually regularize the input data
if(is_regularization)
    I_ref = JLregInputData(I_ref, 5)
end


%% mask image data and GT data
I_ref = I_ref*mask;
fake_fiber= fiber_shape*mask;

cat(3, I_ref, fake_fiber)

%% visualize the sample to see if the OPD is roughly in the estimated range
opdfact = 3.5;%linspace(2,4,20)
lambdaG = 550;
opdmax= opdfact * lambdaG
obj=fake_fiber/max(fake_fiber) * opdmax;

% visualize the fake fiber in RGB
colMix = SimCol(obj);
showCol(colMix)
showCol(I_ref)


%% get histogram
try
    input = double(I_ref);
    hist(reshape(input,[],3),1:max(input(:)));
    colormap([1 0 0; 0 1 0; 0 0 1])
catch
    disp('Error')
end


%% Try out max/sum projection
%obj = sum(obj, [], 1)
%obj = repmat(obj, [size(obj,2), 1])
%Imes = repmat(Imes, [size(Imes,2), 1,1])
%Imes = dip_image(imrotate(double(Imes), -theta_ang));
%Imes = sum(Imes, [], 1)

%% create the LUT for RGB values and their corresponding OPD
[OPDaxis,R,G,B]=JLgetCalibration(obj,I_ref,n_opdsteps);
OPDMap = JLfindOPD(I_ref,R,G,B,OPDaxis,0,1*[1 1 1])

%% Save the RGB - OPD for Tensorflow
matsavepath = strcat(imfolder,'JL_tensorflow.mat');
R_mat = double(R); G_mat = double(G); B_mat = double(B); 
OPD_mat = double(OPDaxis); I_ref_mat = double(I_ref);   
OPDMap_mat = double(OPDMap); mask_mat = double(mask);
save(matsavepath, 'R_mat', 'G_mat', 'B_mat', 'OPD_mat', 'I_ref_mat', 'OPDMap_mat', 'mask_mat' , '-v7.3')


%% do the same thing but now regularized (blurred)
I_ref_reg = JLregInputData(I_ref, 5)
OPDMap_reg = JLfindOPD(I_ref_reg,R,G,B,OPDaxis,0,1*[1 1 1])

%% Try to fit a spline curve to the data and evaluate this as the new RGB values
if(is_splinefit)
    eval_min = 0;
    eval_max = 1200;
    n_samples = 100;
    [R_fit G_fit B_fit, OPDaxis_fit] = JLfitSplineRGB(R, G, B, eval_min, eval_max, n_samples, OPDaxis);
    OPDMap_var = JLfindOPD(I_ref,R_fit,G_fit,B_fit,OPDaxis_fit,0,2*[1 1 1])
end


%%-------------------------------------------------------------------------
% REAL DATA Follows here!
%--------------------------------------------------------------------------


%% try to apply colormap on specific file
I_smpl = dip_image(flip(imread([imfolder, f_sample_im])));
[I_smpl, ROISize, ROICentre] = JLextractROI(I_smpl);
OPDMap_ob = JLfindOPD(I_smpl,R,G,B,OPDaxis,0,0*[1 1 1]);

% display the result
figure
title(f_sample_im)
subplot(121), imagesc(uint8(I_smpl)), axis image, colorbar, title 'RGB Intensity Measurement'
subplot(122), imagesc(double(OPDMap_ob)), axis image, colorbar, colormap gray,  title 'Reconstructed OPD AU'

return
%% load one image and get coordinates
I_smpl = dip_image(flip(imread([sample_folder, sample_files(1).name])));
[I_smpl_sub, ROISize, ROICentre] = JLextractROI(I_smpl);

%% iterate over all files
for ifile = 1:numel(sample_files)
    %% apply to real data
    % extract the ROI
    filepath = [sample_folder, sample_files(ifile).name]
    I_smpl = dip_image(flip(imread(filepath)));
    I_smpl = extract(I_smpl, ROISize, ROICentre);
    
    
    %% remove dust particles
    if(is_dust_particle_remove)
        I_smpl = JLremoveDusti(I_smpl, 5, 51);
    end
    
    %% regularize the input data
    if(is_regularization)
        I_smpl = JLregInputData(I_smpl, 5)
    end
    
    
    %% get back OPD
    OPDMap_ob = JLfindOPD(I_smpl ,R,G,B,OPDaxis,0,0*[1 1 1]);
    
    %% display the result
    figure
    subplot(121), imagesc(uint8(I_smpl)), axis image, colorbar, title 'RGB Intensity Measurement'
    subplot(122), imagesc(double(OPDMap_ob/max(OPDMap_ob(:)))), axis image, colorbar, title 'Reconstructed OPD AU'
end


%%
OPDMap_ob_var_i = dip_image(zeros([size(OPDMap_ob_var, 1)  size(OPDMap_ob_var, 2) 4]));
grayvals = linspace(0, max(OPDMap_ob_var), 5);
for ii = 1:size(grayvals,2)-1
    
    
    g_val_lower = grayvals(ii);
    g_val_upper = grayvals(ii+1);
    
    OPDMap_ob_var_i(:,:,ii-1) = (OPDMap_ob_var>=g_val_lower & OPDMap_ob_var<g_val_upper);
    
end

plot(R_fit)
hold on
plot(G_fit)
plot(B_fit)
plot(OPDaxis_fit)
hold off

cat(3, OPDMap_ob, OPDMap_ob_var, Iob_sub(:,:,1))
OPDMap_ob_var = JLfindOPD(Iob_sub,R_fit,G_fit,B_fit,OPDaxis_fit,0,3*[1 1 1])


I_rgb = double(I_ref);
opd = double(I_ref);
save('ForChristianRGBandOPD', 'I_rgb', 'opd', '-v7.3')

