function output = JLremoveDust(input, Npartikle, box_size)

%% remove dust
% do it for one particle first (test)
myinput = input;

for(i=1:Npartikle)
    
    
    %% create a figure handle and select the coordinates?+?0?oq1    >
    fh=dipshow(myinput(:,:,1)); % find edges of CC signal
    diptruesize(fh, 200);
    %fh=dipshow(abs(AllSubtractFT(:,:,0))^.1) % find edges of CC signal
    fprintf('Please select first dust particle and then the origin of the area which should replace it');
    
    try
        DustPosition = dipgetcoords(fh,2);
        
        gaussian_kernel = exp(-(rr(box_size, box_size)).^2/(box_size));
        
        Area_Dust_Particle  = (1-gaussian_kernel)*extract(myinput, [box_size, box_size], DustPosition(1, :));
        Area_Recover = (gaussian_kernel)*extract(myinput, [box_size,box_size], DustPosition(2, :));
        Area_Result = Area_Recover + Area_Dust_Particle; Area_Result(Area_Result<0)=0;
        
        % select the subroi which has to be exchanged by the new pixelvalues
        replace_width = (DustPosition(1,1)-floor(box_size/2):DustPosition(1,1)+floor(box_size/2));
        replace_height = (DustPosition(1,2)-floor(box_size/2):DustPosition(1,2)+floor(box_size/2));
        
        myinput(replace_width, replace_height,:) = Area_Result;
    catch
        fprintf('an error occured while removing dust')
    end
end


output = myinput;


