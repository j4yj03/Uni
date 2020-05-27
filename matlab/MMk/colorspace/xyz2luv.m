function [ output_image ] = xyz2luv( input_image )
%XYZ2LAB Converte formato CIE XYZ para CIELab 
% Baseado em SZELISKI, R. Computer Vision Algorithms ans Applications.
% Springer-Verlag, 1ª ed, 812p, 2011.
% e
% HUGHES, J. F. et al. Computer Graphics: Principles and Practice. Pearson
% Education, 3ª ed, 1264 p., 2013. 
% Código por Gabriel Maidl (outubro/2017)


% XYZ para o branco nominal RGB = (255,255,255)
Xn = .9505;
Yn = 1;
Zn = 1.088;

im1 = double(input_image)/255;

fY = luv_f(im1(:,:,2)/Yn);

L = 116*fY;

% Valores de u e v para o branco nominal
uw = 4*Xn./(Xn + 15*Yn + 3*Zn);
vw = 9*Yn./(Xn + 15*Yn +3*Zn);

ul = 4*im1(:,:,1)./(im1(:,:,1) + 15*im1(:,:,2) + 3*im1(:,:,3));
vl = 9*im1(:,:,2)./(im1(:,:,1) + 15*im1(:,:,2) +3*im1(:,:,3));

u = 13*L.*(ul - uw);
v = 13*L.*(vl - vw);

output_image(:,:,1) = L;
output_image(:,:,2) = u;
output_image(:,:,3) = v;

% output_image = uint8(output_image*255);

end

function outf = luv_f (t)
delta = 6/29;

outf = zeros(size(t));

index = t>(delta^3);

outf(index) = t(index).^(1/3);

outf(~index) = t(~index)./(3*delta^2) + 2*delta/3;

end