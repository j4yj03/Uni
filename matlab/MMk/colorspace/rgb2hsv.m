function [ output_image ] = rgb2hsv ( input_image )
%RGB2HSV RGB para Hue Saturation Value
% Baseado em AGOSTON, M. K. Computer Graphics and Geometric Modeling:
% Implementation and Angorithms. Springer-Verlag, 1ª ed., 907 p., 2005.
% Entrada: Imagem contendo RGB uint8 0-255
% Saída: HSV contendo valores reais H e S 0-1
% Código por Gabriel Maidl (outubro/2017)

[nL, nC, ~] = size(input_image);

im1 = double(input_image)/255;

R = im1(:,:,1);
G = im1(:,:,2);
B = im1(:,:,3);

maxv = max(max(R,G),B);
minv = min(min(R,G),B);

V = maxv;

S = zeros(nL, nC);
S(maxv>0) = (maxv(maxv>0) - minv(maxv>0))./maxv(maxv>0);

H = zeros(nL, nC);
H(S==0) = -1;

d = maxv - minv;

H(R == maxv) = (G(R == maxv) - B(R == maxv))./d(R == maxv);
H(G == maxv) = 2 + (B(G == maxv) - R(G == maxv))./d(G == maxv);
H(B == maxv) = 4 + (R(B == maxv) - G(B == maxv))./d(B == maxv);

H = H*60;
H(H<0) = H(H<0) + 360;

output_image = zeros([size(H) 3]);

output_image(:, :, 1) = H;
output_image(:, :, 2) = S;
output_image(:, :, 3) = V;

end