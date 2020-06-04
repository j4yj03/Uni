function y = decision(receivedSymbols,constellation) 
% Funktion zur zuordnung der empfangenden Symbole auf einen
% Konstellationspunkt
% EIngabeparameter: die empfangenen Symbole (receivedSymbols) als Vektor,
% sowie die vom Modulationsformat gegebenen Konstellationspunkte;
% Ausgabeparameter: die entschiedenen Symbole (y)
    x = zeros(1,length(receivedSymbols)); % fuer jedes emfangene Symbol existiert ein naehster Konstelationspunkt
    localDistance=zeros(1,length(constellation));    %  Variable, welche den Abstand zwischen Symbol und Konstellationspunkt darstellt
    
    for i=1:1:length(receivedSymbols)   
        for j=1:1:length(constellation)
            localDistance(j) = norm(receivedSymbols(i)-constellation(j)); % jedes empfangene Symbol wird mit jeden moeglichen Konstelationspunkt verglichen
        end
        [val,idx] = min(localDistance); % die kuerzeste Entfernung wird ermittelt
        x(i)=constellation(idx);    % mit dem Index wird dieser Konstellationspunkt vermerkt
    end

    y=x; %Vektor wird uebergeben
end