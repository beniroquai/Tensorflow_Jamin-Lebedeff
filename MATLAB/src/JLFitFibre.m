function [fiber_shape, theta, mask, mybackgroundval] = JLFitFibre(input, is_onesided)
%
% [fiber_shape, theta, mask, mybackgroundval] = JLFitFibre()
%
% IN:
% input - 
% is_onesided - 
%
% OUT:
% fiber_shape - 
% theta - 
% mask - 
% mybackgroundval - 



if(nargin<2)
    is_onesided = false
end
mysize=size(squeeze(input(:,:,1)));
%% Generate a fake fiber object

%% create a figure handle and select the coordinates
%fh=dipshow(input(:,:,1)); % find edges of CC signal
fh=dipshow(showCol(input)); % find edges of CC signal

diptruesize(fh, 200);
%fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
fprintf('Please select 4 coordinates which describe the position of the fiber');
fprintf('Order: Upper-Left Corner, Upper-Right Corner, Lower-Right Corner, Lower-Left Corner \n')
FiberPosition = dipgetcoords(fh,4);
fprintf('Thank you :-)')

% select the Backgroundvalue 
fprintf('Please select a sample-free background region (i.e. black/dark)')
BackgroundPosition = dipgetcoords(fh,1);


mybackgroundval = mean(extract(input, [50,50,3], BackgroundPosition),[],[1,2]);
[fiber_shape, fiber_shape_onesided, theta] = getFakeFiber(FiberPosition, mysize);

%% select ROI
mask = drawpolygon(newim(mysize),FiberPosition,1,'closed');
mask = dip_image(mask,'bin');
mask = ~bpropagation(mask&0,~mask,0,1,1);

if is_onesided
    mask = fiber_shape_onesided>0;
    fiber_shape = fiber_shape_onesided;
end