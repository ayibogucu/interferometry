clear
% addpath("frames")
hologram=imread(['off-axis-data/1.tiff']);
% hologram=rgb2gray(hologram)
hologram=imcrop(hologram)
[N1,N2]=size(hologram);
%area=0.0087/1472*N1*0.0065/1936*N2/400;

lambda=671*10^-9;
z=0.1;
R = 50; 
R0=50;

    
      figure, imshow(flipud(rot90(hologram)), []);
      title('Off-axis hologram')
      %% Calculating Fourier transform and showing absolute value of the result
     spectrum = fftshift(fft2(fftshift(hologram)));
     spectrum_abs = abs(spectrum);
     
      figure, imshow(flipud(rot90(log(spectrum_abs))), []);
      title('Fourier spectrum in log scale / a.u.')
 
%%Blocking the central part of the spectrum
spectrum_abs1 = zeros(N1,N2); 
for ii=1:N1
    for jj=1:N2
     
    x = ii - N1/2;
    y = jj - N2/2;
    
    if (sqrt(x^2 + y^2) > R0) 
        spectrum_abs1(ii, jj) = spectrum_abs(ii,jj); 
    end
    end
end
%% Blocking half of the spectrum
     spectrum_abs1(1:N1/2,:) = 0;   

     
%% Finding the position of the side-band in the spectrum
maximum = max(max(spectrum_abs1));
[x0, y0] = find(spectrum_abs1==maximum)

%% Shifting the complex-valued spectrum to the center
if y0<N2/2
spectrum2 = zeros(N1,N2);
x0 = x0 - N1/2 - 1;
y0 = y0 - N2/2 - 1;
y0=abs(round(y0));
x0=round(x0);
for ii = 1:N1-x0
    for jj = y0+1:N2+y0    
        spectrum2(ii, jj) = spectrum(ii+x0,jj-y0); 
    end
end

end

if y0>N2/2

spectrum2 = zeros(N1,N2);
x0 = x0 - N1/2 - 1;
y0 = y0 - N2/2 - 1;
x0=round(x0);
y0=round(y0);
for ii = 1:N1-x0
    for jj = 1:N2-y0    
        spectrum2(ii, jj) = spectrum(ii+x0,jj+y0); 
    end
end
    
    
    
end
spectrum_abs2=abs(spectrum2);
imshow(log(spectrum_abs2),[])

%% Selecting the central part of the complex-valued spectrum spectrum
spectrum3 = zeros(N1,N2);
for ii=1:N1
    for jj=1:N2
     
    x = ii - N1/2;
    y = jj - N2/2;
    
    if (sqrt(x^2 + y^2) < R) 
        spectrum3(ii, jj) = spectrum2(ii, jj); 
    end
    end
end    
spectrum_abs3=abs(spectrum3);

figure, imshow(flipud(rot90(log(abs(spectrum3)))), []);
      title('Fourier spectrum in log scale / a.u.')
      
      %%
   
     %prop=1;
      reconstruction = ifftshift(ifft2(ifftshift((spectrum3))));
     rec_abs = abs(reconstruction);
     l = angle(reconstruction);


%%
figure, imshow(flipud(rot90(l)), []);
      title('Reconstructed phase wrapped / rad')
      
      s=unwrap_TIE_FFT_DCT(l);
      
      figure
      mesh(s);
%% surface fitting
clear x y

x=1:N1;
y=1:N2;
[xData, yData, zData] = prepareSurfaceData( x, y, s);

% Set up fittype and options.
ft = fittype( 'poly23' );

% Fit model to data.
[fitresult, gof] = fit( [xData, yData], zData, ft );
coef=coeffvalues(fitresult)
clear x y
 parfor x=1:N1
           for y=1:N2
phi2(x,y)    = coef(1) + coef(2)*x + coef(3)*y + coef(4)*x^2 + coef(5)*x*y + coef(6)*y^2 + coef(7)*x^2*y   + coef(8)*x*y^2 + coef(9)*y^3

           end


 end

%% getting results at the end you will have the true scaled mesh of image
 
 result=s-phi2;
 
OPD=-lambda.*result./pi./2;
%OPD=reshape(smooth(OPD),1216,[]);

p_s=4.7;%pixel size
ox=20;%objective magnification
dim1=N2;
dim2=N1;
nc=1.42%random cell ri
nm=1.34;%ranodom medium ri
close all
mesh(linspace(1,8700/ox/1936*N2,dim1),linspace(1,6500/ox/1464*N1,dim2),-OPD/(nc-nm)*10^6);%dimension lengths are 537.6 and 340.5 with pixel size of 5.6 um
%multiplication of 10^6 with OPD is because every dimension is in micro
%scale
 daspect([1 1 1]);
 
 






