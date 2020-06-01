function RGB=readRGB48le( filehandle, width, height )
%Reads 16bps RGB data from rgb48 little endian raw image file
%RGB=readRGB48le( filename, width, height )
%
%Input:
% filname - name of the rgb48 raw image file
% width - width of the RGB image in samples
% height - height of the RGB image in samples
%             
%Output:
% RGB - height x width x color planes uint16 matrix with R,G and B components of the frame

% reads RGB line after line into vector RGB
RGB = fread(filehandle, 3*height*width, 'uint16', 'l');

% sorts each column of the interleaved color plane line vector into
% height number of 3 x width line matrices
RGB = reshape(RGB, 3, width, height);

% permute it to width by height by color planes
RGB = uint16(permute(RGB,[3 2 1]));

end
