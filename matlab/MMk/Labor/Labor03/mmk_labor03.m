%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% MMK SoSe 2019 Laboraufgabe  3 %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Define parameters
% Quantization Parameter
q_scale = 1:5:80; 
% Intra Quantization Matrix
Q_I = [  8, 16, 19, 22, 26, 27, 29, 34;
        16, 16, 22, 24, 27, 29, 34, 37;
        19, 22, 26, 27, 29, 34, 34, 38;
        22, 22, 26, 27, 29, 34, 37, 40;
        22, 26, 27, 29, 32, 35, 40, 48;
        26, 27, 29, 32, 35, 40, 48, 58;
        26, 27, 29, 34, 38, 46, 56, 69;
        27, 29, 35, 38, 46, 56, 69, 83 ];
% zig-zag scan
ZZscan = [  0,  1,  5,  6, 14, 15, 27, 28;
            2,  4,  7, 13, 16, 26, 29, 42;
            3,  8, 12, 17, 25, 30, 41, 43;
            9, 11, 18, 24, 31, 40, 44, 53;
           10, 19, 23, 32, 39, 45, 52, 54;
           20, 22, 33, 38, 46, 51, 55, 60;
           21, 34, 37, 47, 50, 56, 59, 61;
           35, 36, 48, 49, 57, 58, 62, 63 ];
% Preallocation of coefficient vector
QFS=zeros(1,64);   
% figure handles
close all;
forg   =1;
frec   =2;

%% Load Y_OetfBT709_CpBT709_8b (uint8 8bit luma image) from the previous task 2 stored in Y_org.mat
load('Y_org.mat');
% Preallocation of reconstructed picture
Y_rec = uint8(zeros(size(Y_org)));

figure(forg);
imshow(Y_org);
title('Original Y');
    
%% Divide the residual signal Y_res into 8x8 blocks
% TODO: loop over luma 8x8 blocks and for each 8x8 block of Y_org do:
%       1. Apply DCT transform to the 8x8 block of original samples f using the function dct2 to get 
%          transform coefficients F
%i=1;
%j=1;

%Y_rec = zeros(size(Y_org));
D = zeros(length(q_scale));
h=1;
for q = q_scale
    for i = 1:8:size(Y_org,1)
        for j = 1:8:size(Y_org,2)

            f = Y_org(i:i+7, j:j+7);
            %sprintf("%dx%d: @ %d %d",size(f),i,j)
            F = dct2(f);

    %       2. Apply quantization to the coefficients F using the parameter
    %          q_scale and the matrix Q_I (both defined at the beginning) to get quantized
    %          transform coefficients QF
    %TODO

            QF = round((F.*16)./(2*q*Q_I));
            QF(1,1) = round(F(1,1)/8);

    %       3. Apply zig-zag scan to map the 8x8 block QF to the 64x1 vector QFS using
    %          the mapping in the array ZZscan (defined at the beginning)
            for u = 1:8
                for v = 1:8
                    QFS(ZZscan(u,v)+1) = QF(u,v);
                end
            end

    %       4. Apply inverse zig-zag scan to map the 64x1 vector QFS to the 8x8 block QF using
    %          the mapping in the array ZZscan (defined at the beginning)
            for u = 1:8
                for v = 1:8
                    QF(u,v) = QFS(ZZscan(u,v)+1);
                end
            end

    %      5. Apply "inverse quantization" to the quantized coefficients QF using the parameter
    %          q_scale and the matrix Q_I (both defined at the beginning) to get 
    %          reconstructed transform coefficients F_rec
    %TODO
            F = fix((QF.*q.*Q_I)/ 8);
            F(1,1) = QF(1,1)* 8;
    %       6. Apply inverse DCT transform to 8x8 reconstructed transform coefficients F_rec using 
    %          the function idct2 to get the 8x8 reconstructed samples f_rec
            f_rec = idct2(F);

    %       7. Clip the 8x8 reconstructed samples f_rec to 0..255 range and 
    %          copy the 8x8 block to the correct position inside the
    %          reconstructed picture Y_rec
            Y_rec(i:i+7, j:j+7) = uint8(min(max(f_rec, 0), 255));

    % HINT: Start by looping over 8x8 blocks of Y_org and copy each 8x8 block inside the loop
    %       to the correct position inside the reconstructed picture Y_rec.
    %       If that works, the output yuv should correspond to the input yuv
    %       Then, steps 1.-7. can be applied to the 8x8 blocks
        end
    end
    D(h) = psnr(Y_org,Y_rec);
    h=h+1;
end

figure(frec);
imshow(Y_rec);
title('Reconstructed Y');

%% Calculate coded size and distortion of the recunstructed picture
% TODO:
% 1. Calculate the distortion D using the Peak Signal to Noise Ratio PSNR
%    (8bit video!)

%D = psnr(Y_org,Y_rec);

% 2. Calculate and plot D over for various values of q_scale
% q_scale: D     
% 10:      33.67
% 20:      30.28
% 30:      28.59
% 40:      27.47
% 50:      26.67
figure('Name','Werte');
semilogy(q_scale,D,'-r');
legend('PSNR bei verschiedenen qscale');
hold on;
grid on;
grid minor;
