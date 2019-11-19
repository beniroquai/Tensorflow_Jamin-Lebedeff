mysize=[1024,1024];

lambdaG = 520;
opdmax=3 * lambdaG;
obj = 100^2-rr(mysize)^2;
obj(obj<0)=0;
obj=obj/max(obj) * opdmax;

colMix = SimCol(obj);

showCol(colMix)

%%
[OPDaxis,R,G,B]=getCalibration(obj,colMix, 80);

plot(OPDaxis,R,'r')
hold on
plot(OPDaxis,G,'g')
plot(OPDaxis,B,'b')

%% find the right color combination
maxPhotons=1000;
measColImg=noise(colMix/max(colMix) * maxPhotons,'poisson');
showCol(measColImg)
OPDMap = findOPD(measColImg,R,G,B,OPDaxis,[],[1 1 0]);


%%
