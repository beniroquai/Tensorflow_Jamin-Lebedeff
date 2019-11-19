function [OPDaxis,R,G,B]=JLgetCalibration(OPDImg,colMix, NHisto, OPDmax)

if (0)
    [colnormed,CMask]=NormalizeColor(colMix,CThresh);
elseif(0)
    colnormed=colMix/mean(colMix);%, [], [1, 2]);
    sumC = repmat(sum(colMix,[],3),[1 1 3]);
else
    colnormed=colMix;
end

if nargin < 4
    OPDmax = max(OPDImg);
    OPDmax = OPDmax*1.02;
end

OPDaxis = ([0:NHisto]+.5)*OPDmax /NHisto;

%R=zeros(1,NHisto+1);
%G=zeros(1,NHisto+1);
%B=zeros(1,NHisto+1);

i=1;
for n=1:NHisto-1
    OPDval = OPDaxis(n);
    OPDvalNext = OPDaxis(n+1);
    OPDMask = (OPDImg>= OPDval & OPDImg< OPDvalNext);
    if(mean(OPDMask)~=0)
        R{i} = mean(squeeze(colnormed(:,:,0)),OPDMask);
        G{i} = mean(squeeze(colnormed(:,:,1)),OPDMask);
        B{i} = mean(squeeze(colnormed(:,:,2)),OPDMask);
        OPDaxis_temp{i} = OPDaxis(n);
        i = i+1;
    end
end

R = cat(2, R{:});
G = cat(2, G{:});
B = cat(2, B{:});
OPDaxis = cat(2, OPDaxis_temp{:});


