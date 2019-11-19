function [output, ROISize, ROICentre] = JLextractROI(input)
% extracts a ROI by selecting the centre and the outer rim
%% select a sub-roi
%if(~exist('ROIPosition'))
%% create a figure handle and select the coordinates?+?0?oq1    >
%fh=dipshow(input(:,:,1)); % find edges of CC signal
fh=dipshow(showCol(input))
diptruesize(fh, 40);
%fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
fprintf('Please select the center of the ROI you want to process, then click on the outer rim of the ROI.');
ROIPosition = dipgetcoords(fh,2);
ROICentre = ROIPosition(1,:);
ROIOuter = ROIPosition(2,:);
ROIDiff = abs(ROICentre-ROIOuter);
ROIDiff = [max(ROIDiff) max(ROIDiff)];

% display the image
ROISize = 2*ROIDiff;
output = extract(input, ROISize, ROICentre)
