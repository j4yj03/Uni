function T = getMatrixRgb2Xyz( colorspace )
%Determines and calculates matrix to transform RGB with given colorspace into XYZ space
%T = getMatrixRgb2Xyz( colorspace )
%
%Input:
% colorspace -  [optional, default = 'BT709']. Primaries and whitepoint as defined in: 
%             'BT709'         = ITU-R Rec. BT.709 (default)
%             'BT2020'        = ITU-R Rec. BT.2020
%             'ACES'          = Academy Color Encoding Specification (ACES) S-2008-001 or SMPTE ST 2065-1
%             
%Output:
% T - RGB to XYZ transform matrix
%
%Example:
% T = getMatrixRgb2Xyz('BT2020');

disp(['Calculating transform matric for ' colorspace ' RGB to XYZ...']);

if (nargin < 1)
    colorspace = 'BT709';
end

%% TODO
%  define for each colorspace xr, yr, xg, yg, xb, yb, xw, yw
if strcmp(colorspace,'ACES')
    %ACES
    xr = 0.7347;
    yr = 0.2653;
    xg = 0.0000;
    yg = 1.0000;
    xb = 0.0001;
    yb =-0.0770;
    xw =0.32168;
    yw =0.33767;
elseif strcmp(colorspace,'BT709')
    %Rec. ITU-R BT.709
    xr = 0.640;
    yr = 0.330;
    xg = 0.300;
    yg = 0.600;
    xb = 0.150;
    yb =-0.060;
    xw =0.3127;
    yw =0.3290;
elseif strcmp(colorspace,'BT2020')
    %Rec. ITU-R BT.2020
    xr = 0.708;
    yr = 0.292;
    xg = 0.170;
    yg = 0.797;
    xb = 0.191;
    yb =-0.046;
    xw =0.3127;
    yw =0.3290;
else
    error(['Unknown colorspace: ' colorspace]);
end

%% TODO
%  construct matrix A and vector b using xr, yr, xg, yg, xb, yb, xw, yw
A = [xr, xg, xb; yr, yg, yb; (1-xr-yr), (1-xg-yg), (1-xb-yb)];
b = [xw/yw; 1; (1-xw-yw)/yw];
%  calculate vector x using linsolve()
x = linsolve(A,b);
%  construct matrix T using x and xr, xr, xg, yg, xb, yb
%Kr = x(1);
%Kg = x(2);
%Kb = x(3);
%T = [xr*Kr, xg*Kg, xb*Kb;     yr*Kr, yg*Kg, yb*Kb;    (1-xr-yr)*Kr, (1-xg-yg)*Kg, (1-xb-yb)*Kb]

T = A.*x';

end