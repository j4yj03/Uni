function y = demapper(symbols,constellation)
% Funktion zur Umwandlung der empfangenen Symbole in entrepchende Bits
% Eingabeparameter: die entschiedenen Symbole (symbols) als Vektor, die vom
% Modulationsformat gegebenen Konstellationspunkte;
% Ausgabeparameter: die umgewandelten Bits als Vektor (y)
    x=zeros(1,log2(length(constellation))*length(symbols)); % Vektor initialisieren und mit 0 befuellen
    zaehler=1;    % Zaehlervariable, welche den gemappeten Vektor durchlaeuft
    
    for i=1:1:length(symbols)    %fuer alle Symbole
       for j=1:1:length(constellation)    %fuer jeden Konstellationspunkt
           if (symbols(i) == constellation(j))   % sind Symbol und KOnstellationspunkt identisch? 
               bi = de2bi(j-1,floor(log2(length(constellation)-1))+1,'left-msb');    % Dezimalwert in Binaerwert umrechnen, wobei je nach maximal moeglichem Dezimalwert, Bits reserviert werden
               for k=1:1:length(bi)    % fuer alle umgewandelten Bits
                   x(zaehler)=bi(k);    % Eegebnissvektor befuellen
                   zaehler=zaehler+1;
               end
           end
       end
    end

    y=x; %Vektor uebergeben
end