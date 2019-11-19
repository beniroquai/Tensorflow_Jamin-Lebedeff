mysize=[1024,1024];

lambdaG = 520;
opdmax=3 * lambdaG;
obj = 100^2-rr(mysize)^2;
obj(obj<0)=0;
obj=obj/max(obj) * opdmax;

colMix = SimCol(obj);

showCol(colMix)





%% put the 2D image into a 1D vector
num_pixel = size(colMix(:,:,1),1)*size(colMix(:,:,1),2);

pathdiff = reshape(obj, num_pixel , 1); 
pix_r = reshape(colMix(:,:,0), num_pixel , 1);
pix_g = reshape(colMix(:,:,1), num_pixel , 1);
pix_b = reshape(colMix(:,:,2), num_pixel , 1);

%% Convert to double, correlation coeficients (All data)
x=double(pathdiff)'; 

yr=double(pix_r)';  
% Rr=corrcoef(x,yr);

yg=double(pix_g)'; 
% Rg=corrcoef(x,yg);

yb=double(pix_b)'; 
% Rb=corrcoef(x,yb);

%% 6th polynomial by fit (All data)

fr=fit(x,yr,'poly6', 'Exclude', x<0.00001);

fg=fit(x,yg,'poly6', 'Exclude', x<0.00001);

fb=fit(x,yb,'poly6', 'Exclude', x<0.00001);

if(0)
figure('Name','6th polynomial by fit')
scatter(pix_r(0:10:num_pixel), pathdiff(0:10:num_pixel), '.', 'r')
hold on
scatter(pix_g(0:10:num_pixel), pathdiff(0:10:num_pixel), '.', 'g')

scatter(pix_b(0:10:num_pixel), pathdiff(0:10:num_pixel), '.', 'b')
xlabel('Path difference')
ylabel('Colour Intensity')
hold off
end

figure('Name','6th polynomial by fit')
scatter(pix_r(0:10:num_pixel), pathdiff(0:10:num_pixel), '.', 'r')
hold on
scatter(pix_g(0:10:num_pixel), pathdiff(0:10:num_pixel), '.', 'g')

scatter(pix_b(0:10:num_pixel), pathdiff(0:10:num_pixel), '.', 'b')

xlabel('Path difference')
ylabel('Colour Intensity')
hold off
legend('R', 'G', 'B')


%% try to find Calibration data
[OPDaxis,R,G,B]=getCalibration(obj, colMix, 80);

figure;
plot(OPDaxis,R,'r')
hold on
plot(OPDaxis,G,'g')
plot(OPDaxis,B,'b')
hold off

return
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
