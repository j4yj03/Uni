function y=antennaCombining(x, h, combMethod)
% Funktion zur Bestimmung der Linearkombination der einzelnen
% Empfangssignale
% Eingabeparameter: die Antennensignale (x) und die Kanalkoeffizienten (h)
% als Matrix, sowie die Kombinationsmethode (combMethod)
% Ausgabeparameter: Der Vektor der kombinierten Singale (y)

    if strcmp(combMethod,'sum')
        z = ones(size(h));
        
    elseif strcmp(combMethod,'MRC')
        z = conj(h);
        
    elseif strcmp(combMethod,'EGC')
        z = exp(-j*angle(h));
        
    elseif strcmp(combMethod,'SDC')
        %selection diversity r
        %kanalkoeffizienten mit der größten amplitude
        z = max(h);
        
        z_T=zeros(Nr, length(channelCoeff));
            [z_index(1,:),z_index(2,:)]=find(abs(channelCoeff)==max(abs(channelCoeff)));
            for i = 1:1:length(Symbols)
                z_T(z_index(1,i),z_index(2,i))=1;
            end

    else
        error(['Unknown combination method: ' combMethod]);
    end
    y=0;
end