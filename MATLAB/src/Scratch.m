[colim,grayim]=dngShow('C:\Users\marsikovabarbora\Downloads\IMG_20160331_171811.dng');

joinchannels('RGB',colim(:,:,0),colim(:,:,1),colim(:,:,2))

myMat = [0 1 0;1 0 0; 0 0 1]

col2=MatMulDip(myMat,colim);

myMat = [1 -1 1]'

col3=MatMulDip(myMat,colim);

colnormed = colim ./ sum(colim,[],3)

joinchannels('RGB',colnormed(:,:,0),colnormed(:,:,1),colnormed(:,:,2))
