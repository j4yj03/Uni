function [Y,Cb,Cr]=rgb2ycbcr(R,G,B,bitdepth,convmtrx)
%Converts non-linear 4:4:4 R'G'B' in double precision to 
% 4:4:4 YCbCr in integer precision
%[Y,Cb,Cr]=rgb2ycbcr(R,G,B,bitdepth,convmtrx)
%
%Input:
% R,G,B - Non-linear R',G' and B' components in double precision normalized to [0...1]
% bitdepth - bit precision of each sample in Y, Cb and Cr output [optional, default = 8]. 
%            Supported bitdepths are is 8 and 10 
% convmtrx - Conversion matrix [optional, default = 'BT709']. The 
%            following conversions are defined:
%            'BT601' = ITU-R BT.601
%            'BT709' = ITU-R BT.709 (default)
%            'BT2020' = ITU-R BT.2020
%Output:
% Y,Cb,Cr - Non-linear Y',Cb' and Cr' components of the frame in bitdepth 
%           bit precision with restricted range as defined in convmtrx
%
%
%Example:
% [Y,Cb,Cr] = rgb2ybccr(R,G,B,8,'BT709);

if (nargin < 5)
    convmtrx = 'BT709';
end
if (nargin < 4)
    bitdepth = 8;
end

if sum(size(R) ~= size(G)) || sum(size(R) ~= size(B))
    error('Sizes of R, G and B must match!');
end

if bitdepth ~= 8 && bitdepth ~= 10
    error('Bitdepth musst be equal to 8 or 10!');
end

%% TODO
%  define convmtrx specific transform constants Kr and Kb here
if strcmp(convmtrx,'BT601')
    Kr = 0.299;
    Kb = 0.114;
elseif strcmp(convmtrx,'BT709')
    %% TODO
elseif strcmp(convmtrx,'BT2020')
    %% TODO
else
    error(['Unknown conversion matrix: ' convmtrx]);
end

%% TODO
%  do transform here for Y Pb Pr


%% TODO
%  quantize normalized Y Cb Cr based on bitdepth variable either
%  to 8bit values with limited range from 16 to 235 for Y and 16 to 240 for
%  Cb and Cr or
%  to 10bit values with limited range from 64 to 940 for Y and to 16 to 960
%  for CB and Cr

    
%% TODO
%  clip Y Cb Cr output to the bitdepth dependent limited range (see above)
%  and make it uint8() for 8bit and uint16() for 10bit



end