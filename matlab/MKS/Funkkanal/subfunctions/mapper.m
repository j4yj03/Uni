function y = mapper(bits, constellation)
% Funktion zur Zuordnung der Bits zu den entsprechenden Symbolen
% Eingabeparameter: die generierten Bits als Vektor (bits), sowie das
% Konstellationsformat (constellation);
% Ausgabeparameter: die nach dem Konstellationsformat gemappten Symbole

    step = floor(log2(length(constellation)-1))+1; % die Schrittgroesse ist von der Anzahl der Konstellationen abhaengig
    
    if (mod(length(bits),step)~=0)  % Fehlermeldung wenn Bits bei dem Mapvorgang verloren gehen wuerden
        error('Die Anzahl der pro Durchlauf simulierten Bits muss für dieses Modulationsverfahrer durch %d teilbar sein. Derzeit: %d',step,length(bits));
    end
    
    x = zeros(1,floor(length(bits)/step)); % Speicher fuer den Vektor reservieren (wird mit 0en initialisiert)
    
    zaehler=1;    % zweite Zaehlervariable, die den gemappeten Vektor durchlaeuft
    
    for i=1:step:length(bits)-(step-1)
        dec = bi2de(bits(i:i+(step-1)),'left-msb'); % es werden je nach Modulationsverfahren mehrere bits interpretiert
        x(zaehler) = constellation(dec+1);    %Vektor wird an jeder Stelle
        zaehler=zaehler+1;
    end

    y=x;    %Vektor wird uebergeben
end