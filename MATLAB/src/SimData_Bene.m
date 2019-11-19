B=dip_image(imread('/Users/Bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/IMAGES/09_reassembled/100x/IMG_20170404_150510.jpg'));

mysize=size(B, 1); % make it a square 
A{1}=extract(B(:,:,0), [mysize(1) mysize(1)], []);
A{2}=extract(B(:,:,1), [mysize(1) mysize(1)], []);
A{3}=extract(B(:,:,2), [mysize(1) mysize(1)], []);

pixelsize = 85e-6/1000;
mysize=size(A{1});
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
g = q > 8e4 % check for ghost images!!
g_res = real(ift(ft(g)*ft(obj)));
%g_res; %
cat(3, g_res, A{1})

if(0)
% manual selection of the CC-TERM
fh=dipshow(A{1}) % find edges of CC signal
diptruesize(fh,100)
A_coords= dipgetcoords(fh, 4)
else
   
% saved value
A_coords = [1309 1315;1273 1373;489 1285;1037 461];
end


q_artificial = dip_image(zeros(size(obj)));A_coords
q_artificial(sub2ind(size(q_artificial), A_coords(:,2), A_coords(:,1))) = 1;
g_res = real(ift(ft(q_artificial)*ft(obj)));
%g_res; %
cat(3, g_res, A{1})


%% TODO
% only one colourchannel is unsed?
% Convert the RGB value to a unique number and then fit one channel instead of 3

%% put the 2D image into a 1D vector
num_pixel = size(A{1},1)*size(A{1},2);

pathdiff = reshape(g_res, num_pixel , 1); 
pix_r = reshape(A{1}, num_pixel , 1);
pix_g = reshape(A{2}, num_pixel , 1);
pix_b = reshape(A{3}, num_pixel , 1);

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

figure('Name','6th polynomial by fit')
scatter(pathdiff(0:10:num_pixel), pix_r(0:10:num_pixel), '.', 'r')
hold on
plot(fr, 'c')
scatter(pathdiff(0:10:num_pixel), pix_g(0:10:num_pixel), '.', 'g')
plot(fg, 'm')
scatter(pathdiff(0:10:num_pixel), pix_b(0:10:num_pixel), '.', 'b')
plot(fb, 'y')
xlabel('Path difference')
ylabel('Colour Intensity')
xlim([0 0.7])
ylim([0 140])
hold off

%% Crop the area with two beads only, convert to double

g_crop=extract(g_res, [120 150], [1285 1340]);

A1_crop=extract(A{1}, [120 150], [1285 1340]);
A2_crop=extract(A{2}, [120 150], [1285 1340]);
A3_crop=extract(A{3}, [120 150], [1285 1340]);

num_pixel_crop = size(A1_crop,1)*size(A1_crop,2);

pathdiff_crop=reshape(g_crop, num_pixel_crop, 1);
pix_r_crop=reshape(A1_crop, num_pixel_crop, 1);
pix_g_crop=reshape(A2_crop, num_pixel_crop, 1);
pix_b_crop=reshape(A3_crop, num_pixel_crop, 1);

x_crop=double(pathdiff_crop)'; 

yr_crop=double(pix_r_crop)';
yg_crop=double(pix_g_crop)';
yb_crop=double(pix_b_crop)';

%% Croped area two beads- 6th polynomial by fit

fr_crop=fit(x_crop,yr_crop,'poly6', 'Exclude', x_crop<0.00001);

fg_crop=fit(x_crop,yg_crop,'poly6', 'Exclude', x_crop<0.00001);

fb_crop=fit(x_crop,yb_crop,'poly6', 'Exclude', x_crop<0.00001);

figure('Name','Croped area two beads- 6th polynomial by fit')
scatter(x_crop, yr_crop, '.', 'r')
hold on
plot(fr_crop, 'c')
scatter(x_crop, yg_crop, '.', 'g')
plot(fg_crop, 'm')
scatter(x_crop, yb_crop, '.', 'b')
plot(fb_crop, 'y')
xlabel('Path difference')
ylabel('Colour Intensity')
xlim([0 0.7])
ylim([0 140])
hold off

%% Crop the area with one bead only 1, convert to double

g_crop1=extract(g_res, [100 100], [670 1060]);

A1_crop1=extract(A{1}, [100 100], [670 1060]);
A2_crop1=extract(A{2}, [100 100], [670 1060]);
A3_crop1=extract(A{3}, [100 100], [670 1060]);

num_pixel_crop1 = size(A1_crop1,1)*size(A1_crop1,2);

pathdiff_crop1=reshape(g_crop1, num_pixel_crop1, 1);
pix_r_crop1=reshape(A1_crop1, num_pixel_crop1, 1);
pix_g_crop1=reshape(A2_crop1, num_pixel_crop1, 1);
pix_b_crop1=reshape(A3_crop1, num_pixel_crop1, 1);

x_crop1=double(pathdiff_crop1)'; 

yr_crop1=double(pix_r_crop1)';
yg_crop1=double(pix_g_crop1)';
yb_crop1=double(pix_b_crop1)';

%% Croped area one bead only 1 - 6th polynomial by fit

fr_crop1=fit(x_crop1,yr_crop1,'poly6', 'Exclude', x_crop1<0.00001);

fg_crop1=fit(x_crop1,yg_crop1,'poly6', 'Exclude', x_crop1<0.00001);

fb_crop1=fit(x_crop1,yb_crop1,'poly6', 'Exclude', x_crop1<0.00001);

figure('Name','Croped area one bead only 1- 6th polynomial by fit')
scatter(x_crop1, yr_crop1, '.', 'r')
hold on
plot(fr_crop1, 'c')
scatter(x_crop1, yg_crop1, '.', 'g')
plot(fg_crop1, 'm')
scatter(x_crop1, yb_crop1, '.', 'b')
plot(fb_crop1, 'y')
xlabel('Path difference')
ylabel('Colour Intensity')
xlim([0 0.7])
ylim([0 140])
hold off

%% Crop the area with one bead only 2, convert to double

g_crop2=extract(g_res, [90 90], [1040 450]);

A1_crop2=extract(A{1}, [90 90], [1040 450]);
A2_crop2=extract(A{2}, [90 90], [1040 450]);
A3_crop2=extract(A{3}, [90 90], [1040 450]);

num_pixel_crop2 = size(A1_crop2,1)*size(A1_crop2,2);

pathdiff_crop2=reshape(g_crop2, num_pixel_crop2, 1);
pix_r_crop2=reshape(A1_crop2, num_pixel_crop2, 1);
pix_g_crop2=reshape(A2_crop2, num_pixel_crop2, 1);
pix_b_crop2=reshape(A3_crop2, num_pixel_crop2, 1);
% cat(3, g_crop2, A1_crop2, A2_crop2, A3_crop2)

x_crop2=double(pathdiff_crop2)'; 

yr_crop2=double(pix_r_crop2)';
yg_crop2=double(pix_g_crop2)';
yb_crop2=double(pix_b_crop2)';

%% Croped area one bead only 2- 6th polynomial by fit

fr_crop2=fit(x_crop2,yr_crop2,'poly6', 'Exclude', x_crop2<0.00001);

fg_crop2=fit(x_crop2,yg_crop2,'poly6', 'Exclude', x_crop2<0.00001);

fb_crop2=fit(x_crop2,yb_crop2,'poly6', 'Exclude', x_crop2<0.00001);

figure('Name','Croped area one bead only 2- 6th polynomial by fit')
scatter(x_crop2, yr_crop2, '.', 'r')
hold on
plot(fr_crop2, 'c')
scatter(x_crop2, yg_crop2, '.', 'g')
plot(fg_crop2, 'm')
scatter(x_crop2, yb_crop2, '.', 'b')
plot(fb_crop2, 'y')
xlabel('Path difference')
ylabel('Colour Intensity')
xlim([0 0.7])
ylim([0 140])
hold off

%% Crop the area with one bead only 3, convert to double

g_crop3=extract(g_res, [90 90], [490 1270]);

A1_crop3=extract(A{1}, [90 90], [490 1270]);
A2_crop3=extract(A{2}, [90 90], [490 1270]);
A3_crop3=extract(A{3}, [90 90], [490 1270]);

num_pixel_crop3 = size(A1_crop3,1)*size(A1_crop3,2);

pathdiff_crop3=reshape(g_crop3, num_pixel_crop3, 1);
pix_r_crop3=reshape(A1_crop3, num_pixel_crop3, 1);
pix_g_crop3=reshape(A2_crop3, num_pixel_crop3, 1);
pix_b_crop3=reshape(A3_crop3, num_pixel_crop3, 1);

x_crop3=double(pathdiff_crop3)'; 

yr_crop3=double(pix_r_crop3)';
yg_crop3=double(pix_g_crop3)';
yb_crop3=double(pix_b_crop3)';

%% Croped area one bead only 3- 6th polynomial by fit

fr_crop3=fit(x_crop3,yr_crop3,'poly6', 'Exclude', x_crop3<0.00001);

fg_crop3=fit(x_crop3,yg_crop3,'poly6', 'Exclude', x_crop3<0.00001);

fb_crop3=fit(x_crop3,yb_crop3,'poly6', 'Exclude', x_crop3<0.00001);

figure('Name','Croped area one bead only 3- 6th polynomial by fit')
scatter(x_crop3, yr_crop3, '.', 'r')
hold on
plot(fr_crop3, 'c')
scatter(x_crop3, yg_crop3, '.', 'g')
plot(fg_crop3, 'm')
scatter(x_crop3, yb_crop3, '.', 'b')
plot(fb_crop3, 'y')
xlabel('Path difference')
ylabel('Colour Intensity')
xlim([0 0.7])
ylim([0 140])
hold off


fg_crop3_f = feval(fg_crop3,linspace(0, 0.7, 1000));
fr_crop3_f = feval(fr_crop3,linspace(0, 0.7, 1000));
fb_crop3_f = feval(fb_crop3,linspace(0, 0.7, 1000));
figure
plot3(fr_crop3_f, fg_crop3_f, fb_crop3_f)


x_crop1_red = x_crop1(x_crop1>0.001)
yr_crop1_red = yr_crop1(x_crop1>0.001)
figure, scatter(x_crop1_red, yr_crop1_red, '.', 'r')
save('x_y_data', 'x_crop1_red', 'yr_crop1_red', '-v7.3')
