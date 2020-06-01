%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% MMK SoSe 2020 Laboraufgabe  1 %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Read 1920x1080 rgb48le linear RGB image using readRGB48() function 
% into a 3D uint16 array RGB with 16bit per sample
width=1920;
height=1080;
infilename='./Soccer_1080p_ACESLinear_1fr_rgb48le.rgb';
disp(['Open input rgb48le file ' infilename '...']);
fpr = fopen(infilename, 'r');
RGB_ACES = readRGB48le(fpr, width, height);
fclose(fpr);


%% 1. Normalize the 16bit per sample uint16 linear RGB image to RGB image
%     in double precision with values in the [0...1] range
%     and show the RGB image in double precision with imshow()
%% TODO
%  in:  uint16 3D array RGB_ACES (linear ACES)
%  out: double 3D array RGB_ACES (linear ACES)
%% TIPS
%  Use the function double()
RGB_ACES = double(RGB_ACES)./ double(max(max(max(RGB_ACES))));
imshow(RGB_ACES)

%% 2. Transform the double precision linear RGB image in ACES color space
%     into 
%     a) BT.709  RGB colorspace
%     b) BT.2020 RGB colorspace
%     by
%     - calculating transform matrix T_ACES for ACES RGB to XYX
T_ACES = getMatrixRgb2Xyz('ACES');


%     - separating the three components in three 2D matrices R, G and B
R = RGB_ACES(:,:,1);
G = RGB_ACES(:,:,2);
B = RGB_ACES(:,:,3);
%     - transforming the components R, G and B into X, Y and Z using T_ACES

%% TODIIII
X = R.*T_ACES(1,1) + G.*T_ACES(1,2) + B.*T_ACES(1,3);
Y = R.*T_ACES(2,1) + G.*T_ACES(2,2) + B.*T_ACES(2,3);
Z = R.*T_ACES(3,1) + G.*T_ACES(3,2) + B.*T_ACES(3,3);
%%
%     - calculating transform matrix T_BT709 for BT.709 RGB to XYZ
T_BT709 = getMatrixRgb2Xyz('BT709');
%     - calculating the inverse matrix Tinv from T_BT709
Tinv = inv(T_BT709);
%     - transforming the components X, Y and Z into R, G and B using Tinv

R = X.*Tinv (1,1) + Y.*Tinv (1,2) + Z.*Tinv (1,3);
G = X.*Tinv (2,1) + Y.*Tinv (2,2) + Z.*Tinv (2,3);
B = X.*Tinv (3,1) + Y.*Tinv (3,2) + Z.*Tinv (3,3);
%     - merging R, G, and B in one 3D array RGB_CpBT709
RGB_CpBT709 = zeros(size(RGB_ACES));
RGB_CpBT709(:,:,1)=R;
RGB_CpBT709(:,:,2)=G;
RGB_CpBT709(:,:,3)=B;

%     - displaying RGB_CpBT709 using imshow()
imshow(RGB_CpBT709)
%%
%     - calculating transform matrix T_BT2020 for BT.2020 RGB to XYZ
T_BT2020 = getMatrixRgb2Xyz('BT2020');
%     - calculating the inverse matrix Tinv from T_BT2020
Tinv = inv(T_BT2020);
%     - transforming the components X, Y and Z into R, G and B using Tinv
R = X.*Tinv (1,1) + Y.*Tinv (1,2) + Z.*Tinv (1,3);
G = X.*Tinv (2,1) + Y.*Tinv (2,2) + Z.*Tinv (2,3);
B = X.*Tinv (3,1) + Y.*Tinv (3,2) + Z.*Tinv (3,3);
%     - merging R, G, and B in one 3D array RGB_CpBT2020
RGB_CpBT2020 = zeros(size(RGB_ACES));
RGB_CpBT2020(:,:,1)=R;
RGB_CpBT2020(:,:,2)=G;
RGB_CpBT2020(:,:,3)=B;
%     - displaying RGB_CpBT2020 using imshow()
imshow(RGB_CpBT2020)
%% TODO 
%  in:  double 3D array RGB_ACES (linear ACES)
%  out: double 3D array RGB_CpBT709 (linear BT.709)
%       double 3D array RGB_CpBT2020 (linear BT.2020)
%% TIPS
%  Fill the function skeleton in getMatrixRgb2Xyz.m for all three
%  colorspaces, call the function like this:
%    T_ACES = getMatrixRgb2Xyz('ACES');
%    T_BT709 = getMatrixRgb2Xyz('BT709');
%    T_BT2020 = getMatrixRgb2Xyz('BT2020');
%  The inverse of a matrix can be calculated by the function inv()
%% 3. Transform the double precision linear RGB image into double precision 
%     non-linear RGBn image using BT.709 OETF by
%     - applying BT.709 OETF separately for each component of RGB_CpBT709
Rn=oetf(RGB_CpBT709(:,:,1),'BT709');
Gn=oetf(RGB_CpBT709(:,:,2),'BT709');
Bn=oetf(RGB_CpBT709(:,:,3),'BT709');
%     - merging Rn, Gn, and Bn(non linear) in one 3D array RGBn_OetfBT709_CpBT709
RGBn_OetfBT709_CpBT709 = zeros(size(RGB_ACES));
RGBn_OetfBT709_CpBT709(:,:,1)=Rn;
RGBn_OetfBT709_CpBT709(:,:,2)=Gn;
RGBn_OetfBT709_CpBT709(:,:,3)=Bn;
%     - displaying RGBn_OetfBT709_CpBT709 using imshow()
imshow(RGBn_OetfBT709_CpBT709)
%%
%     - applying BT.709 OETF separately for each component of RGB_CpBT2020
Rn=oetf(RGB_CpBT2020(:,:,1),'BT709');
Gn=oetf(RGB_CpBT2020(:,:,2),'BT709');
Bn=oetf(RGB_CpBT2020(:,:,3),'BT709');
%     - merging Rn, Gn, and Bn in one 3D array RGBn_OetfBT709_CpBT2020
RGBn_OetfBT709_CpBT2020 = zeros(size(RGB_ACES));
RGBn_OetfBT709_CpBT2020(:,:,1)=Rn;
RGBn_OetfBT709_CpBT2020(:,:,2)=Gn;
RGBn_OetfBT709_CpBT2020(:,:,3)=Bn;
%     - displaying RGBn_OetfBT709_CpBT2020 using imshow()
imshow(RGBn_OetfBT709_CpBT2020)
%% TODO 
%  in:  double 3D array RGB_CpBT709 (linear BT.709)
%       double 3D array RGB_CpBT2020 (linear BT.2020)
%  out: double 3D array RGBn_OetfBT709_CpBT709 (non-linear BT.709)
%       double 3D array RGBn_OetfBT709_CpBT2020 (non-linear BT.2020)
%% TIPS
%  Fill the function skeleton in oetf.m, call the function for eahc
%  component like this:

