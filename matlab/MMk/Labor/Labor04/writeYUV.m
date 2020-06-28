function writeYUV( Y, U, V, format, filehandle, frame )

if strcmp(format,'yuv420p10le')   
   type='*uint16';
elseif strcmp(format,'yuv420p')
   type='*uint8';
else
    error(['Unknown yuv format: ' format ' Valid formats are: yuv420p and yuv420p10...']); 
end

disp(['Write ' format ' frame ' num2str(frame) '...']);

fwrite(filehandle, Y', type, 'l');
fwrite(filehandle, U', type, 'l');
fwrite(filehandle, V', type, 'l');

end
