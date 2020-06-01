function RGB = showrgb(RGB)

%create color image showing red component from black (0,0,0) to red (255,0,0)
Rcolor=uint8(zeros(size(RGB)));
Rcolor(:,:,1)=RGB(:,:,1);
%create color image showing green component from black (0,0,0) to green (0,255,0)
Gcolor=uint8(zeros(size(RGB)));
Gcolor(:,:,2)=RGB(:,:,2);
%create color image showing blue component from black (0,0,0) to blue (0,0,255)
Bcolor=uint8(zeros(size(RGB)));
Bcolor(:,:,3)=RGB(:,:,3);

%calculate histogram for each color component
histo=zeros(3,256);
for comp=1:size(RGB,3)
    for i=1:256
        histo(comp,i)=sum(sum(RGB(:,:,comp)==(i-1)));
    end
end

%create figure with 2x4 subplots
figure;
% RGB image at 1
subplot(2,4,1);
imshow(RGB);
title('RGB');
% R image at 2
subplot(2,4,2);
imshow(Rcolor);
title('R');
% G image at 3
subplot(2,4,3);
imshow(Gcolor);
title('G');
% B image at 4
subplot(2,4,4);
imshow(Bcolor);
title('B');

%R histogram at 6
subplot(2,4,6);
bar(histo(1,:),'r');
set(gca,'XLim',[1 256]);
set(gca,'YLim',[0 3500]);

%R histogram at 7
subplot(2,4,7);
bar(histo(2,:),'g');
set(gca,'XLim',[1 256]);
set(gca,'YLim',[0 3500]);

%R histogram at 8
subplot(2,4,8);
bar(histo(3,:),'b');
set(gca,'XLim',[1 256]);
set(gca,'YLim',[0 3500]);




