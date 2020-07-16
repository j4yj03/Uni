%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% MMK SoSe 2019 Laboraufgabe  4 %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define parameters
% Input YUV
width=352;
height=288;
infilename='foreman_352x288_5fr_8bit.yuv';
fpr_org = fopen(infilename, 'rb');
disp(['Open input yuv420p file ' infilename '...']);
frame=0;
eof=false;
% Quantization Parameter
q_scale = 10; 
% MPEG1 Intra Quantization Matrix
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
% Preallocation of picture arrays
% coefficient vector
QFS=zeros(1,64);
% original picture
Y_org = zeros(height,width);     
% prediction picture
Y_pred = Y_org;
% residual picture
Y_res = Y_org;
% reconstructed residual picture
Y_recres = Y_org;
% reconstructed picture
Y_rec = Y_org;
U_rec = 128*ones(height/2,width/2); 
V_rec = 128*ones(height/2,width/2);
% Output YUV
outfilename = ['foreman_352x288_5fr_8bit_qp' num2str(q_scale) '.yuv'];
fpw_rec = fopen(outfilename, 'wb');
disp(['Open output yuv420p file ' infilename '...']);
% figure handles
close all;
forg   =1;
fpred  =2;
fres   =3;
frecres=4;
frec   =5;


%% loop over frames
while ~eof
    % read 352x288 8bit YUV image into three uint8 arrays Y, U, V
    [Y_org, U_org, V_org, eof] = readYUV420p(fpr_org, width, height);
    if eof
        break;
    end
    disp('------------------------------------------------------');
    disp(['Read frame ' num2str(frame) '...']);

    % Intra-/Inter-picture Prediction
    figure(forg);
    imshow(Y_org/255);
    title('Original Y');
    pause;
    if frame == 0
        % a) INTRA: For the first frame do intra prediction
        %    -> set the prediction signal Y_pred to 0
        Y_pred = zeros(height,width);
    else
        % b) INTER: For all other frames do inter prediction
        %    -> set the prediction signal Y_pred to the previous reference frame Y_ref
        Y_pred = Y_ref;     
    end
    %show prediction
    figure(fpred);
    imshow(uint8(Y_pred));
    title('Prediction Y');
    pause;
    % calculate residual signal Y_res (prediction error) by subtracting
    % the prediction signal Y_pred from the original signal Y_org
    Y_res=Y_org-double(Y_pred);
    %show residual signal
    figure(fres);
    imshow((Y_res+255)/510);
    title('Residual Y');
    pause;
    
    %% TODO:
    % - Add loop from Labor03
     for i = 1:8:size(Y_res,1)
        for j = 1:8:size(Y_res,2)

            f = Y_res(i:i+7, j:j+7);
            %sprintf("%dx%d: @ %d %d",size(f),i,j)
            F = dct2(f);

            QF = round((F.*16)./(2*q_scale.*Q_I));
            QF(1,1) = round(F(1,1)/8);

    %       zig-zag scan
            for u = 1:8
                for v = 1:8
                    QFS(ZZscan(u,v)+1) = QF(u,v);
                end
            end

    %       inverse zig-zag scan
            for u = 1:8
                for v = 1:8
                    QF(u,v) = QFS(ZZscan(u,v)+1);
                end
            end

            F = fix((QF.*q_scale.*Q_I)/ 8);
            F(1,1) = QF(1,1)* 8;
            f_rec = idct2(F);
            
            Y_recres(i:i+7, j:j+7) = min(max(f_rec, -255), 255);
        end
    end
    
    % - Inside the loop
    %    a) replace at the beginning before 1.: Y_org by Y_res    
    %    b) remove in 7.: clipping f_rec to 0..255 range 
    %       because now the residual (Y_org-Y_pred) is procssed which
    %       ranges from -255 to 255
    %    c) replace at the end in 7.: Y_rec by Y_recres
    %% HINT:
    % You can watch the yuv vraw video files using YUView:
    % https://github.com/IENT/YUView/releases

    % Updated inner loop from Labor03:
%       1. Apply DCT transform to the 8x8 block of residual samples f using the function dct2 to get 
%          transform coefficients F
%       2. Apply quantization to the coefficients F using the parameter
%          q_scale and the matrix Q_I (both defined at the beginning) to get quantized
%          transform coefficients QF
%       3. Apply zig-zag scan to map the 8x8 block QF to the 64x1 vector QFS using
%          the mapping in the array ZZscan (defined at the beginning)
%       4. Apply inverse zig-zag scan to map the 64x1 vector QFS to the 8x8 block QF using
%          the mapping in the array ZZscan (defined at the beginning)   
%       5. Apply "inverse quantization" to the quantized coefficients QF using the parameter
%          q_scale and the matrix Q_I (both defined at the beginning) to get 
%          reconstructed transform coefficients F_rec
%       6. Apply inverse DCT transform to 8x8 reconstructed transform coefficients F_rec using 
%          the function idct2 to get the 8x8 reconstructed samples f_rec
%       7. Copy the 8x8 block to the correct position inside the
%          reconstructed residual picture Y_recrec

    figure(frecres);
    imshow((Y_recres+255)/510);
    title('Reconstructed Residual Y');
    pause;
    
    % add prediction to get reconstruction
    Y_rec = double(Y_pred)+Y_recres;
    
    % clip to [0,255] range (8bit)
    Y_rec = uint8(Y_rec);
    figure(frec);
    imshow(Y_rec);
    title('Reconstructed Y');
    pause;
    
    % save reconstructed arrays for next frame inter-picture prediction
    Y_ref = Y_rec;
    
    % write out reconstructed arrays into YUV file
    writeYUV(Y_rec,U_rec,V_rec,'yuv420p',fpw_rec,frame);
    frame=frame+1;
end
fclose(fpr_org);
fclose(fpw_rec);