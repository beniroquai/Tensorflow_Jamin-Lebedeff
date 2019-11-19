save('testconstraintopd', 'I_ref', 'R', 'G', 'B')
load('testconstraintopd')
RGBImg=extract(I_ref, [256 256]);
%% Try nearest neighbour search
mydict = squeeze(double([R', G', B']));
mymeasR = RGBImg(:,:,0);
mymeasG = RGBImg(:,:,1);
mymeasB = RGBImg(:,:,2);
mymeasRGB = double(cat(2,mymeasR(:),mymeasG(:),mymeasB(:)));
% find minimum distance to k-nearest RGB-datapoints in the look-up-table
[myindex, mydistance] = knnsearch(mydict,mymeasRGB','k',10,'distance','euclidean');

figure, plot3(squeeze(double(R)),squeeze(double(G)),squeeze(double(B)))%, 15, double(squeeze(OPDaxis))), xlabel 'R', xlabel 'G', xlabel 'B', title ' parametric function OPD(RGB)' 
figure, plot3(squeeze(double(mymeasR)),squeeze(double(mymeasG)),squeeze(double(mymeasB)))
% reconstruct the image from the nearest neighbors
myopd = zeros(size(RGBImg));
myres = {};
for(ii = 1:size(myindex,2))
    myopd(1:size(myindex,1)) = OPDaxis(myindex(:,ii));
    myres{ii}=sum(dip_image(myopd),[],3);
end

% now we have a stack of images with degrading quality - but: Eventually we
% can find a mix of images which minimizes the total variation of the
% phase-gradient 
phase_candidates = cat(3, myres{:});
cat(3, RGBImg, phase_candidates)


mymask = dilation(erosion(imfill(double(canny(squeeze(phase_candidates(:,:,0)))), 'holes'),2),2);
cat(3,mymask, squeeze(phase_candidates(:,:,0)))
%% generate masks for gradients
if(0)
    mysize=size2d(RGBImg);
    aa=newim(mysize);
    aa(mysize(1)/2, mysize(2)/2) = 1;
    bb{ii}=aa;
    mymask = {};
    for ii=1:10%mysize(1)/2
        bb{ii+1}=dilation(aa,ii*2,'rectangular');
        mymask {ii}=bb{ii+1}-bb{ii};
    end
    mymask = cat(3,mymask {:});
end

% Mask the different colors 
grayImage = double(squeeze(phase_candidates(:,:,0)));
numberOfClasses = 5;
indexes = kmeans(grayImage(:), numberOfClasses);
classImage = reshape(indexes, size(grayImage));
cat(3,dip_image(classImage), squeeze(phase_candidates(:,:,0)))




%% compute gradients in x and y
mytv_radial={};
mydx_pix = 1;
mydx = (size(phase_candidates,2)-mydx_pix)/size(phase_candidates,2);
for ii=0:size(phase_candidates,3)-1
    mytv_radial{ii+1}=squeeze(phase_candidates(:,:,ii) - affine_trans(squeeze(phase_candidates(:,:,ii)), [mydx mydx], [0 0], 0))^2;
end
mytv_radial=cat(3,mytv_radial{:})

%%
mygradx = dx(phase_candidates,3);
mygrady = dy(phase_candidates,3);
tvvar = sqrt(mygradx^2+mygrady^2)

%% Now get the pixel value with least variation
% We now need to select the pixel in the stack with least total variation
tvvar * mymask(:,:,1)
[minval,minidx]=min(double(tvvar),[],3);
tvvar_mat = double(tvvar);
[rows, columns] = ndgrid(1:size(tvvar_mat, 1), 1:size(tvvar_mat, 2));


% there might be a simpler trick I'm not aware of..
mymask = zeros(size(tvvar_mat)); 
for ii=1:size(tvvar_mat,3)
    mymask_tmp = zeros(size2d(tvvar_mat));
    mymask(:,:,ii) = dip_image(minidx==ii);
end
myresult = dip_image(sum(mymask*phase_candidates, [], 3));
