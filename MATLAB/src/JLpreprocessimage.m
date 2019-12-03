function I_out = JLpreprocessimage(I_in,mygausskerneldim)
I_in = I_in/2^8;
%I_in = log(1+I_in);
I_in(:,:,0) = gaussf(I_in(:,:,0),mygausskerneldim);
I_in(:,:,1) = gaussf(I_in(:,:,1),mygausskerneldim);
I_in(:,:,2) = gaussf(I_in(:,:,2),mygausskerneldim);
I_out = I_in;
