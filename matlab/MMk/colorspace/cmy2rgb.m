function [ output_image ] = cmy2rgb( input_image )
%RGB2CMY Converte formato CMY para RGB 
% HUGHES, J. F. et al. Computer Graphics: Principles and Practice. Pearson
% Education, 3ª ed, 1264 p., 2013. 
% Código por Gabriel Maidl (outubro/2017)

im1 = double(input_image)/255;

output_image = uint8(1-im1)*255;

end