function rawImg=getRawImg16(Img)

raw=dngRead(Img);

% R G
% G B

M = size(raw, 2);
N = size(raw, 1);

red_mask = uint16(repmat([1 0; 0 0], M/2, N/2));
green_mask = uint16(repmat([0 1; 1 0], M/2, N/2));
blue_mask = uint16(repmat([0 0; 0 1], M/2, N/2));

R=uint16(raw.*red_mask);
G=uint16(raw.*green_mask);
B=uint16(raw.*blue_mask);

R( ~any(R,2), : ) = [];
R( :, ~any(R,1) ) = [];
R=dip_image(R);

G1=uint16(repmat([0 1; 0 0], M/2, N/2)).*G;
G1( ~any(G1,2), : ) = [];
G1( :, ~any(G1,1) ) = [];
G1=dip_image(G1);
G2=uint16(repmat([0 0; 1 0], M/2, N/2)).*G;
G2( ~any(G2,2), : ) = [];
G2( :, ~any(G2,1) ) = [];
G2=dip_image(G2);
G=(G1+G2)./2;

B( ~any(B,2), : ) = [];
B( :, ~any(B,1) ) = [];
B=dip_image(B);

rawImg=cat(3,R,G,B);

end


