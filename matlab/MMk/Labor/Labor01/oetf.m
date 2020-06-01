function En = oetf( E, f )
%Applies an electro-optical transfer function to a linear luminance
% En = oetf( E, f)
%
%Input:
% E - linear luminance
% f -  [optional, default = 'BT709']. Supported OETFs are: 
%             'BT709' = Non linear transfer function as defined in ITU-R Rec. BT.709 (default)
%             
%Output:
% En - non-linear luminance
%

if (nargin < 2)
    f = 'BT709';
end

[height, width] = size(E);
En = zeros(height, width);

if strcmp(f,'BT709')
    %% siehe skript S61 maskierungsmatrix
    %% TODO 
    if (E < 0.018)
        En = (E.*4.5);
    else
        En = (E.^0.45*1.099-0.099);
    end
    
else
    error(['Unknown eotf: ' f]);
end

end