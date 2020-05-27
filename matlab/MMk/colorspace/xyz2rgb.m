function [ output_image ] = xyz2rgb( input_image )
%XYZ2RGB Converte formato CIE XYZ para RGB 
% Baseado em SZELISKI, R. Computer Vision Algorithms ans Applications.
% Springer-Verlag, 1ª ed, 812p, 2011.
%
% Código por Gabriel Maidl (outubro/2017)

% inv(matriz para RGB709)
const = [3.24048134320053 -1.53715151627132 -0.498536326168888;
    -0.969254949996568 1.87599000148989 0.0415559265582928;
    0.0556466391351772 -0.204041338366511 1.05731106964534];

%[R; G; B] = const*[X; Y; Z];

[nL, nC, ~] = size(input_image);

im1 = double(input_image)/255;

output_image = zeros(nL, nC, 3);

for l1 = 1:nL
    for c1 = 1:nC
        rgb_pixel = reshape(im1(l1, c1, :), [3,1,1]); % Contrói [X; Y; Z;]
        output_image(l1, c1, :) = const*rgb_pixel;
    end
end

output_image = uint8(output_image*255);

end

