function colMix=SimCol(obj)
lambdaB = 450;
lambdaG = 520;
lambdaR = 630;

lambdaB = 390;
lambdaG = 570;
lambdaR = 630;


B = sin(pi*obj/lambdaB)^2;
G = sin(pi*obj/lambdaG)^2;
R = sin(pi*obj/lambdaR)^2;

colPure = cat(3,R,G,B);
% showCol(colPure)

MixMat = [1 0.2 0.2;0.15 1 0; 0.1 0.4 1];
colMix = MatMulDip(MixMat,colPure);
