% RemoveHotPixels(anImage, treshValue, kernelSize): Removes cosmic rays events from Raman image. 
% 
% Identify the hot pixels in the image and then remove them - replace them with median surroundings.  
% 
% anImage: Image
% treshValue: The limit for hot pixels that are allowed in the final image.  
% kernelSize: The size of kernel used by medif function. If not given, uses default (7). 

function img=RemoveHotPixels(anImage, treshValue, kernelSize)
if nargin<1 || isempty(anImage)
    anImage=imread('orka.tif');
end
if nargin<3 || isempty(kernelSize)
    kernelSize=7;
end

img=anImage;

value=max(max(img));
mymask=img>value-1; 
img1=medif(img,kernelSize); 
img(mymask)=img1(mymask);  


value=max(max(img));
mymask2=img>value-1; 
tresh=sum(sum(mymask2)); 
if tresh>treshValue
    disp('failed removing all the hot pixels'); 
    imshow(img);
else imshow(img);
end
end