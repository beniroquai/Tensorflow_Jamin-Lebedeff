function OPDMap = findOPD(RGBImg,R,G,B,OPDaxis,CThresh,GaussSigma)
if nargin < 6 || isempty(CThresh)
    CThresh = 0.001;
end
if nargin < 7
    GaussSigma = 0;
end

if (0)
    [colnormed,CMask]=NormalizeColor(RGBImg,CThresh);
else
    colnormed=RGBImg /mean(RGBImg);
    sumC = repmat(sum(RGBImg,[],3),[1 1 3]);
    CMask = sumC > CThresh;
end

% calculate the error for all combinations
R = reshape(dip_image(R),[1 1 size(R,2)]);
G = reshape(dip_image(G),[1 1 size(G,2)]);
B = reshape(dip_image(B),[1 1 size(B,2)]);
myErr = (colnormed(:,:,0) - R)^2 + (colnormed(:,:,1) - G)^2 + (colnormed(:,:,2) - B)^2;
if any(GaussSigma)>0
    myErr=gaussf(myErr,GaussSigma);
end

CMask = repmat(CMask(:,:,0),[1 1 size(R,3)]);
[minv,minp]=min(myErr,CMask,3);

OPDaxis=dip_image(OPDaxis);
OPDMap=minp*0.0;
OPDMap(:)=OPDaxis(double(minp(:)));
