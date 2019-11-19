function [output]=extracBayerChannel(im)

% example:
% testim = readim('IMG_20170621_160615.dng')
% extractBayerChannel(testim)
%
%
% [output] = extractBayerChannel(im)
% Bilinear Interpolation of the missing pixels
% Bayer CFA
%       G R
%       B G
% Output = a complete RGB image on 3 channels

M = size(im, 1);
N = size(im, 2);

red_mask = dip_image(uint8(repmat([0 1; 0 0], N/2, M/2)));
green_mask = dip_image(uint8(repmat([1 0; 0 1], N/2, M/2)));
blue_mask = dip_image(uint8(repmat([0 0; 1 0], N/2, M/2)));

R=uint8(im.*red_mask);
G=uint8(im.*green_mask);
B=uint8(im.*blue_mask);

G= G + imfilter(G, [0 1 0; 1 0 1; 0 1 0]/4); % green at R and B

B1 = imfilter(B,[0 1 0; 0 0 0; 0 1 0]/4); % blue at R
B2 = imfilter(B+B1,[1 0 1; 0 1 0; 1 0 1]/4); % blue at G
B = B + B1 + B2;

R1 = imfilter(R,[0 0 0; 1 0 1; 0 0 0]/4); % red at B
R2 = imfilter(R+R1,[1 0 1; 0 1 0; 1 0 1]/4); % red at G
R = R + R1 + R2;


output(:,:,1)=R; output(:,:,2)=G; output(:,:,3)=B;
end