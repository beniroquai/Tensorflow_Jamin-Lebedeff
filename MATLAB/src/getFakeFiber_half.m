function [fiber_shape, alpha_avg]= getFakeFiber_half(FiberPosition, mysize)
%% get angle of the fiber laying in the sample
% in dip-coordinates the first is X, wheras in Matlab it's y-coordinate
alpha_left = atan((FiberPosition(1,2)-FiberPosition(4,2))/(FiberPosition(1,1)-FiberPosition(4,1)));
alpha_right = atan((FiberPosition(2,2)-FiberPosition(3,2))/(FiberPosition(2,1)-FiberPosition(3,1)));
alpha_avg = mean([alpha_left, alpha_right]);

center_left = FiberPosition(1,:)+(FiberPosition(4,:)-FiberPosition(1,:))/2;
center_right = FiberPosition(2,:)+(FiberPosition(3,:)-FiberPosition(2,:))/2;

% calculate the diameter of the fiber (Pythagoras)
dist_left_right = sqrt((center_left(1)-center_right(1)).^2+(center_left(2)-center_right(2)).^2);

% calculate the central position of the fiber
center_fiber = center_left+(center_right-center_left)/2;

% debug: Show line
length_line = 400;

x1 = -sin(alpha_avg+pi/2)*length_line+center_fiber(1);
y1 = +cos(alpha_avg+pi/2)*length_line+center_fiber(2);

x2 = sin(alpha_avg+pi/2)*length_line+center_fiber(1);
y2 = -cos(alpha_avg+pi/2)*length_line+center_fiber(2);
if(0)
    line([x1,x2],[y1,y2])
end



%% draw a fiber
center_fiber_xy = round(center_fiber-mysize/2);
xy_grid = -cos(alpha_avg)*(yy(mysize)-center_fiber_xy(2))+sin(alpha_avg)*(xx(mysize)-center_fiber_xy(1));
%xy_grid = circshift((xy_grid), round(center_fiber-size(I_subtractR)/2))
fiber_shape = real(sqrt((dist_left_right/2).^2-(xy_grid).^2));
