mysize=[1024,1024];

lambdaG = 520;
opdmax=3 * lambdaG;
obj = 100^2-rr(mysize)^2;
obj(obj<0)=0;
obj=obj/max(obj) * opdmax;

colMix = SimCol(obj);

showCol(colMix)

%% Assign data from experiment
if(1)
    A=dip_image(imread('/Users/Bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/IMAGES/IMG_20170404_150510_square.jpg'));
    
    if(0)%size(B, 1)==size(B, 0))
        mysize=min(size(B, 1), size(A, 0)); % make it a square
        A{1}=extract(B(:,:,0), [mysize(1) mysize(1)], []);
        A{2}=extract(B(:,:,1), [mysize(1) mysize(1)], []);
        A{3}=extract(B(:,:,2), [mysize(1) mysize(1)], []);
    end
    
    
    
    % Define Constants
    pixelsize = 85e-6/1000;
    pixelsize = .5e-6/1000;
    mysize=[size(A{1},1), size(A{1},2)];
    mysize_p = mysize*pixelsize;
    
    %% Generate the "fake" sphere object
    lambdaG = 550;
    opdmax=3 * lambdaG;
    radius_sphere = 4e-6; % ???
    obj = radius_sphere^2-pixelsize*rr(mysize,'freq')^2;
    obj(obj<0)=0;
    obj=obj/max(obj) * opdmax;
    
    %% Place ideal sphere everywhere a real-one is expected
    crosscorr_res = real(ift(conj(ft(obj)).*ft(A{1})));
    q = maxima(crosscorr_res, 2)*crosscorr_res;
    g = q > 4e5; % check for ghost images!!
    g_res = real(ift(ft(g)*ft(obj)));
    g_res; %
    
    obj = squeeze(g_res(:,:,0));
    colMix = A;
    cat(3, obj, colMix)
    
end

%mysize=[size(B, 1), size(B, 2)];

%colMix = dip_image(zeros(mysize(1), mysize(2),3));
%colMix(:,:,0) = A{1};
%colMix(:,:,1) = A{2};
%colMix(:,:,2) = A{3};

%% try to find Calibration data
[OPDaxis,R,G,B]=getCalibration(obj, colMix, 80);

figure;
plot(OPDaxis,R,'r')
hold on
plot(OPDaxis,G,'g')
plot(OPDaxis,B,'b')
hold off

%
% save('RGB_OPDAxis', 'OPDaxis', 'R', 'G', 'B', '-v7.3')
% % interpolate data to get more samples
% NHistoq = 2000;
% OPDmax = 1560;
% OPDaxisq = ([0:NHistoq]+0.5)*OPDmax /NHistoq;
% Rq = interp1(OPDaxis,R,OPDaxisq,'spline');
% % Save data to have training data for the network
% x_crop1_red = OPDaxisq(1:1800);
% yr_crop1_red = Rq(1:1800);
% save('x_y_data', 'x_crop1_red', 'yr_crop1_red', '-v7.3')
%
% plot(OPDaxis,R,'r')
% hold on
% plot(OPDaxis,G,'g')
% plot(OPDaxis,B,'b')


%% find the right color combination
maxPhotons=1000;
measColImg=noise(colMix/max(colMix) * maxPhotons,'poisson');
showCol(measColImg)

measColImg = colMix;
OPDMap = findOPD(measColImg,R,G,B,OPDaxis,[],[1 1 0])

%%
