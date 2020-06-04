function y = generateBits(x) 
% Funktion zur gleichverteilten Erzeugung von Bits
% Eingabeparameter: die Anzahl der zu erzeugenden Bits (x);
% Ausgabeparameter (y): Anzahl der erzeugten Bits

    if (x <= 0) || ( x >= 2e15) 
        error('groesse des Vektors darf nicht kleiner als 0 und nicht groesser als 2^15');
    end
    
    y = randi([0 1], 1,x);    % Bit-Vektor erzeugen

end