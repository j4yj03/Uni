function [ output_image ] = xyz2lab( input_image )
%XYZ2LAB Converte formato CIE XYZ para CIELab 
% Baseado em SZELISKI, R. Computer Vision Algorithms ans Applications.
% Springer-Verlag, 1ª ed, 812p, 2011.
%
% Código por Gabriel Maidl (outubro/2017)

% XYZ para o branco nominal RGB = (255,255,255)
Xn = .9505;
Yn = 1;
Zn = 1.088;

im1 = double(input_image)/255;

fY = lab_f(im1(:,:,2)/Yn);

L = 116*fY;

a = 500*(lab_f(im1(:,:, 1)/Xn) - fY);

b = 200*(fY - lab_f(im1(:,:, 3)/Zn));

output_image(:,:,1) = L;
output_image(:,:,2) = a;
output_image(:,:,3) = b;

% output_image = uint8(output_image*255);

end

function outf = lab_f (t)
delta = 6/29;

outf = zeros(size(t));

index = t>(delta^3);

outf(index) = t(index).^(1/3);

outf(~index) = t(~index)./(3*delta^2) + 2*delta/3;

end