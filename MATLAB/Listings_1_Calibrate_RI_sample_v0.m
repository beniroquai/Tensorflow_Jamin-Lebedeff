%% This file tries to recover the OPD of a sample with known reference
% It takes two cellphone images, where one is a known fiber-like object
% Authors: B. Diederich, B. Marsikova, B. Amos, R. Heintzmann
close all

%% Add some parameters here
is_dust_particle_remove = false; % do you want to remove dust particles in the ref. image?
is_regularization = false; % eventually regularize the input image (blur it)
is_raw = false; % is te file a RAW-dng? Experimental since each phone handles it a little different!
is_onesided = true; % tis parameter is useful if there is a gradient in the images visible
n_opdsteps = 50;

% add the library
addpath('./src')

% determine the path for the images
imfolder = './data/EXP3/'
f_ref_img = 'IMG_20191125_142818_PS2.jpg';
f_sample_im = 'IMG_20191125_143408_PS2.jpg'; % PS stands for DNG-> JPG using Potosop RAW converter

% load the imagefiles
if(is_raw)
    I_ref_raw = flip(uint16(dngRead([imfolder, f_ref_img_raw])));
    I_ref_raw = extracBayerChannel(I_ref_raw);
    I_ref_raw = dip_image(I_ref_raw)/2^8;
else
    I_ref_raw = (dip_image(flip(imread([imfolder, f_ref_img])))/2^8);
end


%% extract the Ref.-ROI
I_ref = JLextractROI(I_ref_raw);
mysize=size(squeeze(I_ref(:,:,1)));

%% remove dust particles
if(is_dust_particle_remove)
    I_ref = JLremoveDust(I_ref, 0, 51);
end

%% fit the artificial fiber in the experimental dataset
[fiber_shape, theta, mask, mybackgroundval] = JLFitFibre(I_ref, is_onesided);

% Visualize the groundtruth and its measurement
cat(3, I_ref , fiber_shape)

%% Eventually regularize the input data
if(is_regularization)
    I_ref = JLregInputData(I_ref, 5)
end


%% mask image data and GT data
I_ref_masked = I_ref*mask;
fake_fiber= fiber_shape*mask;

cat(3, I_ref, fake_fiber)

%% visualize the sample to see if the OPD is roughly in the estimated range
opdfact = 3.5;%linspace(2,4,20)
lambdaG = 550;
opdmax= opdfact * lambdaG
obj=fake_fiber/max(fake_fiber);

% visualize the fake fiber in RGB
colMix = SimCol(obj);
showCol(colMix)
showCol(I_ref)


%% create the LUT for RGB values and their corresponding OPD
I_ref_masked_var = log(1+I_ref_masked);
[OPDaxis,R,G,B]=JLgetCalibration(obj,(I_ref_masked_var),n_opdsteps,mybackgroundval);
OPDMap = JLfindOPD((I_ref_masked_var),R,G,B,OPDaxis,0,1*[1 1 1])

% Save the RGB - OPD for Tensorflow
matsavepath = strcat(imfolder,'JL_tensorflow.mat');
R_mat = double(R); G_mat = double(G); B_mat = double(B);
OPD_mat = double(OPDaxis); I_ref_mat = double(I_ref_masked);
OPDMap_mat = double(OPDMap); mask_mat = double(mask);
OPDBackgroundval_mat = double(mybackgroundval);
save(matsavepath, 'R_mat', 'G_mat', 'B_mat', 'OPD_mat', 'I_ref_mat', 'OPDMap_mat', 'mask_mat', 'opdmax', 'OPDBackgroundval_mat', '-v7.3')
%%
figure()
subplot(121)
plot(OPD_mat,R_mat), hold on
plot(OPD_mat,G), plot(OPD_mat,B)
legend('r','g','b')
hold off
subplot(122)
plot3(R_mat, G_mat, B_mat)
xlabel('R')
ylabel('G')
zlabel('B')

%% s-------------------------------------------------------------------------
% REAL DATA Follows here!
%--------------------------------------------------------------------------


% try to apply colormap on specific file
I_smpl = (dip_image(flip(imread([imfolder, f_sample_im])))/2^8);
I_smpl = log(1+I_smpl );
[I_smpl, ROISize, ROICentre] = JLextractROI(I_smpl);

OPDMap_ob = JLfindOPD(I_smpl,R,G,B,OPDaxis,0,0*[1 1 1]);

% display the result
figure
title(f_sample_im)
subplot(121), imagesc(double(I_smpl)), axis image, colorbar, title 'RGB Intensity Measurement'
subplot(122), imagesc(double(OPDMap_ob)), axis image, colorbar, colormap gray,  title 'Reconstructed OPD AU'




