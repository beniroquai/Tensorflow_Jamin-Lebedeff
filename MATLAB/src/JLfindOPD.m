function OPDMap = JLfindOPD(RGBImg,R,G,B,OPDaxis,CThresh,GaussSigma)
if nargin < 6 || isempty(CThresh)
    CThresh = 0.001;
end
if nargin < 7
    GaussSigma = 0;
end

if (0)
    [colnormed,CMask]=NormalizeColor(RGBImg,CThresh);
elseif(0)
    colnormed=RGBImg /mean(RGBImg);
    sumC = repmat(sum(RGBImg,[],3),[1 1 3]);
    CMask = sumC > CThresh;
else
    CMask = 0*RGBImg;
    colnormed=RGBImg ;
end

% calculate the error for all combinations
R = reshape(dip_image(R),[1 1 size(R,2)]);
G = reshape(dip_image(G),[1 1 size(G,2)]);
B = reshape(dip_image(B),[1 1 size(B,2)]);
if(0)
    % try to gather the near-by regions of the parametric plot
    figure
    plot(squeeze(double(R))), hold on, plot(squeeze(double(G))), plot(squeeze(double(B))), hold off, legend 'R, G, B'
    plot3(squeeze(double(R)), squeeze(double(G)), squeeze(double(B)))
    scatter3(squeeze(double(R)), squeeze(double(G)), squeeze(double(B)), squeeze(double(OPDaxis)))
    
end
% Measure minimum L2 distance to closest Colorvalue
myErr = (colnormed(:,:,0) - R)^2 + (colnormed(:,:,1) - G)^2 + (colnormed(:,:,2) - B)^2;
if any(GaussSigma)>0
    myErr=gaussf(myErr,GaussSigma*0);
end


[minv,minp]=min(myErr, [],3);


%scatter3(squeeze(double(R)), squeeze(double(G)), squeeze(double(B)), 19, squeeze(double(OPDaxis)))
%%
% Trying to implment some kind of TV-regularisation
OPDaxis = dip_image(OPDaxis);
OPDMap=minp*0.0;
OPDMap(:)= (double(minp(:)));
OPDMap = squeeze(OPDMap)/size(OPDaxis);
diffx = (OPDMap-circshift(OPDMap, [0 1]))^2;
diffy = (OPDMap-circshift(OPDMap, [1 0]))^2;
cat(3, diffx, diffy);
