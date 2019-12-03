function [OPDaxis,R,G,B]=JLgetCalibration(OPDImg,colMix,NHisto, mybackgroundval)

if nargin<4
    mybackgroundval = [0,0,0];
end

if(1)
    % linear scaling of the OPD-classes
    OPDaxis = linspace(0.01,1,NHisto);
else
    % maybe better to have parabolic scaling of the OPD classes to have
    % better discretization of the steps (e.g. same number of points per
    % class
    OPDaxis = linspace(0.01,1,NHisto).^.5;
end

%OPDImg = resample(OPDImg, [2,2]);
%colMix = resample(colMix, [2,2,1]);
%% iterate over all OPD "classes"
i=1;
mytemp = {};
%OPDImg = 1-(sqrt(1-OPDImg^2)); % we want linearly seperated opds 

% Add background-value from previously selected region:
R{1} = mybackgroundval(0);
G{1} = mybackgroundval(1);
B{1} = mybackgroundval(2);
OPDaxis_temp{1}  = 0;

for n=1:NHisto-1
    OPDval = OPDaxis(n);
    OPDvalNext = OPDaxis(n+1);
    OPDMask = (OPDImg>= OPDval & OPDImg < OPDvalNext);
    mytemp{n} = OPDMask;
    if(mean(OPDImg*OPDMask)~=0) % only take those values which have an OPD value assigned
        i = i+1;  
        R{i} = mean(squeeze(colMix(:,:,0)),OPDMask);
        G{i} = mean(squeeze(colMix(:,:,1)),OPDMask);
        B{i} = mean(squeeze(colMix(:,:,2)),OPDMask);
        myopdval = (OPDImg*OPDMask);
        %mean(myopdval(myopdval>0));
        %std(myopdval(myopdval>0))
        OPDaxis_temp{i} = mean(myopdval(myopdval>0));
          
    end
end
cat(4,mytemp{:})

        
% cast all values
R = cat(2, R{:});
G = cat(2, G{:});
B = cat(2, B{:});
OPDaxis = dip_image(real(cat(2, OPDaxis_temp{:})));


