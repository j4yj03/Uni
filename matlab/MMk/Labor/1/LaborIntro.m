%close all open figures
close all;

% read image baboon.png
A = imread('baboon.png');
% store three color planes in A in three matrices R, G, B
R = A(:,:,1);
G = A(:,:,2);
B = A(:,:,3);

%show min and max values of R
min(min(R))
max(max(R))

%show whole image A
figure;
imshow(A);

%show R, G and B
figure;
imshow(R);
figure;
imshow(G);
figure;
imshow(B);

%show first 256x256 quarter of A
figure;
imshow(A(1:256,1:256,:));

%create 4 subplots into one figure
figure;
subplot(1,4,1);
imshow(A);
title('RGB');
subplot(1,4,2);
imshow(R);
title('R');
subplot(1,4,3);
imshow(G);
title('G');
subplot(1,4,4);
imshow(B);
title('B');

%color R, G and B images by converting it to a RGB image with the other
%two components being all 0
Rcolor=uint8(zeros(size(A)));
Rcolor(:,:,1)=R;
Gcolor=uint8(zeros(size(A)));
Gcolor(:,:,2)=G;
Bcolor=uint8(zeros(size(A)));
Bcolor(:,:,3)=B;
figure;
subplot(1,4,1);
imshow(A);
title('RGB');
subplot(1,4,2);
imshow(Rcolor);
title('R');
subplot(1,4,3);
imshow(Gcolor);
title('G');
subplot(1,4,4);
imshow(Bcolor);
title('B');

%adjust brightness and contrast of the image using the function
%adjustimage in adjustimage.m
% increase brightness by 50% (of max value 256)
brightness = 128;
contrast   = 0;
A2 = adjustimage(A,brightness,contrast);
% increase contrast by 50% (of max value 256)
brightness = 0;
contrast   = 128;
A3 = adjustimage(A,brightness,contrast);
% decrease brightness by 50% (of min value -256)
brightness = -128;
contrast   = 0;
A4 = adjustimage(A,brightness,contrast);
% decrease contrast by 50% (of min value -256)
brightness = 0;
contrast   = -128;
A5 = adjustimage(A,brightness,contrast);

% make 5 plots showing R,G,B and histogram for each image A, A2, A3, A4 and A5
% using showrgb in showrgb.m:
for img={A A2 A3 A4 A5}
    showrgb(img{1});
end

