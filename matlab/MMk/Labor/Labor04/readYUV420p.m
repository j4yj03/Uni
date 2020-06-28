function [Y, U, V, eof] = readYUV420p(fpr, width, height)

Y = zeros(height,width);     
U = zeros(height/2,width/2);
V = zeros(height/2,width/2);
eof=false;

for m=1:height
    temp=fread(fpr, width);
    if isempty(temp)
        eof=true;
        break;
    end
    Y(m,:)=temp;
end
if isempty(temp)
    eof=true;
    return;
end
for m=1:(height/2)
    U(m,:) = fread(fpr, width/2);
end
for m=1:(height/2)
    V(m,:) = fread(fpr, width/2);
end  

end