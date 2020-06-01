function RGB=adjustimage(RGB,brightness,contrast)

a = (259*(contrast+255)) / (255*(259-contrast));
RGB = uint8(a*(double(RGB)-128)+128+brightness);
