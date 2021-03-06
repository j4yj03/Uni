function [R,G,B]=ycbcr2rgb(Y,Cb,Cr,bitdepth,convmtrx)
%Converts YCbCr to RGB
%[R,G,B]=ycbcr2rgb(Y,Cb,Cr,bitdepth,convmtrx,fullrange)
%
%Input:
% Y,Cb,Cr - Non-linear Y',Cb' and Cr' components of the frame in bitdepth bit precision
% bitdepth - bitdepth bit precision of each sample in Y, Cb and Cr [optional, default = 8]. Supported 
%            bitdepth is 8 bits 
% convmtrx - Conversion matrix [optional, default = 'BT601']. The 
%            following conversions ase defined:
%            'BT601' = ITU-R BT.601
%            'BT709' = ITU-R BT.709
%            'BT2020' = ITU-R BT.2020
%Output:
% R,G,B - Non linear R',G' and B' components of the frame in double
% precision with full range from 0.0 to 1.0.
%
%
%Example:
% [R,G,B]=ycbcr2rgb(Y,Cb,Cr,8,'BT709');

if (nargin < 5)
    convmtrx = 'BT601';
end
if (nargin < 4)
    bitdepth = 8;
end

if sum(size(Y) ~= size(Cb)) || sum(size(Cb) ~= size(Cr))
    error('Sizes of Y, Cb and Cr must match!');
end

if bitdepth ~= 8
    error('Bitdepth musst be equal to 8!');
end

%% TODO
%  define convmtrx specific transform constants Kr and Kb here
if strcmp(convmtrx,'BT601')
    Kr = 0.299;
    Kb = 0.114;
elseif strcmp(convmtrx,'BT709')
    Kr = 0.2126;
    Kb = 0.0722;
elseif strcmp(convmtrx,'BT2020')
    Kr = 0.2627;
    Kb = 0.0593;
else
    error(['Unknown conversion matrix: ' convmtrx]);
end


%% TODO
%  normalize quantized Y Cb Cr from 8bit values with limited range 
%  from 16 to 235 for Y and 16 to 240 for Cb and Cr 
%  to double precision values from 0.0 to 1.0 for Y and 
%  from -0.5 to 0.5 for Pb and Pr
%  HINT: convert Y Cb Cr component to double precision using double() first


Y = double(Y - 16);
Cb = double(Cb - 16);
Cr = double(Cr - 16);

%Yt = Y./ max(max(Y));
%Pb = Cb./ max(max(Cb)) - 0.5;
%Pr = Cr./ max(max(Cr)) - 0.5;


Yt = Y./ 219;
Pb = (Cb./ 224) - 0.5;
Pr = (Cr./ 224) - 0.5;
%RGB_ACES = double(RGB_ACES)./ double(max(max(max(RGB_ACES))));


%%


R = 2 * Pr - 2 * Kr * Pr + Yt;
B = 2 * Pb - 2 * Kb * Pb + Yt;
G = (B * Kb + Kr * R - Yt) / (Kb + Kr - 1);



%% TODO    
%  clip R G B output
R = double(min(max(R, 0.0), 1.0));
G = double(min(max(G, 0.0), 1.0));
B = double(min(max(B, 0.0), 1.0));

end