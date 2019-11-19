function [fiber_shape, theta, mask] = JLFitFibre_half(input)

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

[fiber_shape, theta] = getFakeFiber(FiberPosition, mysize);

%% divide fiber in two pieces 
center_left = FiberPosition(1,:) + (FiberPosition(2,:)-FiberPosition(1,:))/2
center_right = FiberPosition(4,:) + (FiberPosition(3,:)-FiberPosition(4,:))/2

FiberPosition(2,:) = center_left;
FiberPosition(3,:) = center_right;

%% select ROI
mask = drawpolygon(newim(mysize),FiberPosition,1,'closed');
mask = dip_image(mask,'bin');
mask = ~bpropagation(mask&0,~mask,0,1,1);

fiber_shape = fiber_shape*mask;