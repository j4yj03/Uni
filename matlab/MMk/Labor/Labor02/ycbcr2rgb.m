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
    %% TODO
elseif strcmp(convmtrx,'BT709')
    %% TODO
elseif strcmp(convmtrx,'BT2020')
    %% TODO
else
    error(['Unknown conversion matrix: ' convmtrx]);
end


%% TODO
%  normalize quantized Y Cb Cr from 8bit values with limited range 
%  from 16 to 235 for Y and 16 to 240 for Cb and Cr 
%  to double precision values from 0.0 to 1.0 for Y and 
%  from -0.5 to 0.5 for Pb and Pr
%  HINT: convert Y Cb Cr component to double precision using double() first


%% TODO
%  do transform here for R G B with full range

%% TODO    
%  clip R G B output


end