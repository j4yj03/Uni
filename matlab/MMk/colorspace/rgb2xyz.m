function [ output_image ] = rgb2xyz( input_image )
%RGB2XYZ Converte formato RGB para CIE XYZ
% Baseado em SZELISKI, R. Computer Vision Algorithms ans Applications.
% Springer-Verlag, 1ª ed, 812p, 2011.
%
% Código por Gabriel Maidl (outubro/2017)

%matriz para RGB709
const = [0.412453 0.357580 0.180423;
0.212671 0.715160 0.072169;
0.019334 0.119193 0.950227];

%[X; Y; Z] = const*[R; G; B];

[nL, nC, ~] = size(input_image);

im1 = double(input_image)/255;

output_image = zeros(nL, nC, 3);

for l1 = 1:nL
    for c1 = 1:nC
        rgb_pixel = reshape(im1(l1, c1, :), [3,1,1]); % Contrói [R; G; B;]
        output_image(l1, c1, :) = const*rgb_pixel;
    end
end

output_image = uint8(output_image*255);

end

