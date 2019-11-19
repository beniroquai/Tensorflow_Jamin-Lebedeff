function [colnormed,CMask]=NormalizeColor(RGBImg,CThresh)
if nargin < 2 || isempty(CThresh)
    CThresh = 0.1;
end

sumC = repmat(sum(RGBImg,[],3),[1 1 3]);
CMask = sumC > CThresh;
colnormed =0 .* RGBImg;
colnormed(CMask) = RGBImg(CMask) ./ sumC(CMask);
