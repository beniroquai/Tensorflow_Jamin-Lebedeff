function [OPDaxis,R,G,B]=getCalibration(OPDImg,colMix, NHisto, CThresh, OPDmax)
if nargin < 4  || isempty(CThresh)
    CThresh = 0.1;
end

if (0)
    [colnormed,CMask]=NormalizeColor(colMix,CThresh);
else
    colnormed=colMix/mean(colMix);%, [], [1, 2]);
    sumC = repmat(sum(colMix,[],3),[1 1 3]);
    CMask = sumC > CThresh;
end

if nargin < 5
    OPDmax = max(OPDImg,squeeze(CMask(:,:,0)));
    OPDmax = OPDmax*1.02;
end

OPDaxis = ([0:NHisto]+0.5)*OPDmax /NHisto;

R=zeros(1,NHisto+1);
G=zeros(1,NHisto+1);
B=zeros(1,NHisto+1);
for n=1:NHisto-1
    OPDval = OPDaxis(n);
    OPDvalNext = OPDaxis(n+1);
    OPDMask = squeeze(CMask(:,:,0)) & (OPDImg>= OPDval & OPDImg< OPDvalNext);
    R(n) = mean(squeeze(colnormed(:,:,0)),OPDMask);
    G(n) = mean(squeeze(colnormed(:,:,1)),OPDMask);
    B(n) = mean(squeeze(colnormed(:,:,2)),OPDMask);
end


