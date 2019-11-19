function Imes = JLregInputData(Imes, method)
% try to manipulate the input data by regularizing it
switch method
    case  1
        
        Imes(:,:,0) = gaussf(Imes(:,:,0), 1);
        Imes(:,:,1) = gaussf(Imes(:,:,1), 1);
        Imes(:,:,2) = gaussf(Imes(:,:,2), 1);
    case  2
        len = 500;
        theta_ang = -theta/pi*180;
        PSF = fspecial('motion', len, theta_ang);
        PSF = (PSF>0)*mean(mean(PSF));
        
        Imes(:,:,0) = dip_image(imfilter(double(Imes(:,:,0)), PSF, 'circular'));
        Imes(:,:,1) = dip_image(imfilter(double(Imes(:,:,1)), PSF, 'circular'));
        Imes(:,:,2) = dip_image(imfilter(double(Imes(:,:,2)), PSF, 'circular'));
    case  3
        Imes = Imes/mean(Imes, [], [1,2]);
    case  4
        Imes(:,:,0) = gaussf_adap(squeeze(Imes(:,:,0)),[],[2 0],0,0);
        Imes(:,:,1) = gaussf_adap(squeeze(Imes(:,:,1)),[],[2 0],0,0);
        Imes(:,:,2) = gaussf_adap(squeeze(Imes(:,:,2)),[],[2 0],0,0);
    case  5
        Imes(:,:,0) = wiener2(double(squeeze(Imes(:,:,0))),[5 5]);
        Imes(:,:,1) = wiener2(double(squeeze(Imes(:,:,1))),[5 5]);
        Imes(:,:,2) = wiener2(double(squeeze(Imes(:,:,2))),[5 5]);
        Imes = dip_image(Imes);
end
