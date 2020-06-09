%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% MMK SoSe 2020 Laboraufgabe  2 %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load RGBn_* (non-linear double precision images) from the previous task 1 stored in RGBn.mat
load('RGBn.mat');

%% 1. Transform the non-linear RGB images to YCbCr using the corresponding 
%     RGB to YCbCr transform and quantize 
%     a) RGBn_OetfBT709_CpBT709 to 8bit
%     b) RGBn_OetfBT709_CpBT2020 to 10bit
%  TODO: fill the function skeleton in rgb2ycbcr.m
%
disp('Applying RGB to 8bit YCbCr conversion for BT709 OETF BT709 colorspace RGB image...');
[Y_OetfBT709_CpBT709_8b,   Cb_OetfBT709_CpBT709_8b,   Cr_OetfBT709_CpBT709_8b]   = rgb2ycbcr(RGBn_OetfBT709_CpBT709(:,:,1), RGBn_OetfBT709_CpBT709(:,:,2), RGBn_OetfBT709_CpBT709(:,:,3),  8,'BT709');
%
disp('Applying RGB to 10bit YCbCr conversion for BT709 OETF BT2020 colorspace RGB image...');
[Y_OetfBT709_CpBT2020_10b,  Cb_OetfBT709_CpBT2020_10b,  Cr_OetfBT709_CpBT2020_10b]  = rgb2ycbcr(RGBn_OetfBT709_CpBT2020(:,:,1), RGBn_OetfBT709_CpBT2020(:,:,2), RGBn_OetfBT709_CpBT2020(:,:,3), 10,'BT2020');

%% 2. Apply 4:2:0 color-subsampling schemes to the 4:4:4 8bit YCbCr image 
%  HINT: use the function imresize()
Y = Y_OetfBT709_CpBT709_8b;
%Cb = Cb_OetfBT709_CpBT709_8b;
%Cr = Cr_OetfBT709_CpBT709_8b;
Cb = imresize(Cb_OetfBT709_CpBT709_8b, 0.5);
Cr = imresize(Cr_OetfBT709_CpBT709_8b, 0.5);

%% 3. Display each 8bit Y, Cb and Cr component in a separate figure using imshow()
%
subplot(3,1,1), imshow(Y)
subplot(3,1,2), imshow(Cb)
subplot(3,1,3), imshow(Cr)

%YCbCr = zeros(size(RGBn_OetfBT709_CpBT709));
%YCbCr(:,:,1) = Y;
%YCbCr(:,:,2) = Cb;
%YCbCr(:,:,3) = Cr;

%imshow(YCbCr)

%% 4. Calculate the size in bytes of the 4:4:4 RGB 8bit array and the 
%     4:2:0 YCbCr 8bit array when 8bits=1byte per sample are used.
%
sizebit444 = size(RGBn_OetfBT709_CpBT709,1) * size(RGBn_OetfBT709_CpBT709,2) * size(RGBn_OetfBT709_CpBT709,3);
sizebyte444 = sizebit444/8;
sizekbyte444 = sizebyte444/1024;

sizebit420 = size(Y,1) * size(Y,2) + size(Cb,1) * size(Cb,2) + size(Cr,1) * size(Cr,2);
sizebyte420 = sizebit420/8;
sizekbyte420 = sizebyte420/1024;

sprintf("4:4:4 = "+sizekbyte444+" kbyte\n4:2:0 = "+sizekbyte420+" kbyte")

%% 5. Upsample the 4:2:0 Y Cb Cr image back to 4:4:4 Y Cb Cr
%
%  HINT: use the function imresize()

%Y = Y;
Cb = imresize(Cb, 2);
Cr = imresize(Cr, 2);

%% 6. Convert the 4:4:4 Y Cb Cr image back to 4:4:4 R G B by implementing the inverse 
%     transform in a new function similar to rgb2ycbcr()
%
[R,G,B]=ycbcr2rgb(Y,Cb,Cr,8,'BT709');

%%
imshow(RGBn_OetfBT709_CpBT709)
%% 7. Compare the original RGB image RGBn_OetfBT709_CpBT709 with the reconstructed 
%      RGB image consisting of R, G, B by displaying them side-by-side

RGBn = zeros(size(RGBn_OetfBT709_CpBT709));
RGB(:,:,1)=R;
RGB(:,:,2)=G;
RGB(:,:,3)=B;

imshow(RGB)