close all

% load file
imfolder = '/Users/Bene/Desktop/OpenCamera/';

% look for all images in folder
imagefileDNG = 'IMG_20171124_144041.jpg';
%Iav_raw = dip_image((getRawImg16([imfolder, imagefileDNG])))
Iav_raw = (dip_image(flip(imread([imfolder, imagefileDNG]))));
Iav_raw = Iav_raw.^2;
% extract the ROI
Iav = JLextractROI(Iav_raw);
mysize=size(squeeze(Iav(:,:,1)));


% remove dust particles
Iav_test = JLremoveDust(Iav, 0, 51);

% try to normalize the colourchannels
%I_subtract = Iav_test/mean(Iav_test, [], [1,2]);

% fit fiber into the acquired data
[fiber_shape, theta, mask] = JLFitFibre(Iav_test);

% Visualize the groundtruth and its measurement
cat(3, Iav_test , fiber_shape)




%% Experiment with the separation of the colourchannels
if(0)
    Imes = I_mask*Iav_test;
    Imesbluemean = mean(Imes);
    Imes(Imes>210) = Imesbluemean ;
    Imesblue = Imes(:,:,2);
    Imesbluemin = min(Imesblue(I_mask));
    Imesblue = (Imesblue-Imesbluemin);
    Imes(:,:,2) = Imesblue*5;
    Imes(Imes>250) = Imesbluemean ;
else
    Imes = Iav_test;
end

% regularize the input data
Imes = JLregInputData(Imes, 5)

% mask image data and GT data
Iav_test = Iav_test*mask;
Imes = Imes*mask;
fake_fiber= fiber_shape*mask;

cat(3, Iav_test, Imes)

%% visualize the sample to see if the OPD is roughly in the estimated range
opdfact = 2;%linspace(2,4,20)
lambdaG = 550;
opdmax= opdfact * lambdaG
obj=fake_fiber/max(fake_fiber) * opdmax;

% visualize the fake fiber in RGB
colMix = SimCol(obj);
showCol(colMix)



%% get histogram
try
    input = double(Imes);
    hist(reshape(input,[],3),1:max(input(:)));
    colormap([1 0 0; 0 1 0; 0 0 1])
catch
end


%% Try out max/sum projection
%obj = sum(obj, [], 1)
%obj = repmat(obj, [size(obj,2), 1])
%Imes = repmat(Imes, [size(Imes,2), 1,1])
%Imes = dip_image(imrotate(double(Imes), -theta_ang));
%Imes = sum(Imes, [], 1)

%% create the LUT for RGB values and their corresponding OPD
[OPDaxis,R,G,B]=JLgetCalibration(obj,Imes,50);
OPDMap = JLfindOPD(Imes,R,G,B,OPDaxis,0,1*[1 1 1])


%% Try to fit a spline curve to the data and evaluate this as the new RGB values
eval_min = 0;
eval_max = 1200;
n_samples = 100;
[R_fit G_fit B_fit, OPDaxis_fit] = JLfitSplineRGB(R, G, B, eval_min, eval_max, n_samples, OPDaxis);
OPDMap_var = JLfindOPD(Imes,R_fit,G_fit,B_fit,OPDaxis_fit,0,2*[1 1 1])

%


%% apply to real data 
imagefileDNG_obj = 'IMG_20171124_145359.jpg';
imagefileDNG_obj = 'IMG_20171124_150643.jpg';
Iob_file_name = [imfolder imagefileDNG_obj];
%Iob = dip_image(getRawImg16(Iob_file_name));
Iob = dip_image(imread(Iob_file_name));
Iob = Iob.^2;
% extract the ROI
Iob_sub = JLextractROI(Iob);

% remove dust particles
if(0)
Iob_sub = JLremoveDusti(Iob_sub, 5, 51);
end

% regularize the input data
Iob_sub = JLregInputData(Iob_sub, 5)


% get back OPD
OPDMap_ob = JLfindOPD(Iob_sub ,R,G,B,OPDaxis,0,0*[1 1 1]);
OPDMap_ob_var = JLfindOPD(Iob_sub ,R_fit,G_fit,B_fit,OPDaxis_fit,0,[1 1 1]);

 
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


I_rgb = double(Iav_test);
opd = double(Iav_test);
save('ForChristianRGBandOPD', 'I_rgb', 'opd', '-v7.3')

