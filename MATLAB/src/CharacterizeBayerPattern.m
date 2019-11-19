ffolder = '/Users/Bene/Dropbox/Dokumente/Promotion/PROJECTS/Jamin.Lebedeff/IMAGES/Calibration/OpenCamera/'
dngfiles = dir([ffolder, '*.dng']);  

nfiles = length(dngfiles);    % Number of files found
for ii=1:nfiles
   currentfilename = dngfiles(ii).name;
   [colim,grayim]=dngShow([ffolder, currentfilename]);
   % currentimage = dngRead([ffolder, currentfilename])
   
   % joinchannels('RGB',colim(:,:,0),colim(:,:,1),colim(:,:,2))

   crop_image = extract(colim, [300, 300]);
   R_val{ii} = mean(crop_image(:,:,0), []);
   G_val{ii} = mean(crop_image(:,:,1), []);
   B_val{ii} = mean(crop_image(:,:,2), []);
end

lambdas = linspace(400, 700, nfiles)
figure
plot(lambdas, cat(2, R_val{:})', 'r')
hold on
plot(lambdas, cat(2, G_val{:})', 'g')
plot(lambdas, cat(2, B_val{:})', 'b')
hold off

legend('red-channel', 'green-channel', 'blue-channel')
xlabel('lambda [nm]')
ylabel('counts [au]')