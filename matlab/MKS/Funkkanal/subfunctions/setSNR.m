function y=setSNR(x, snrlin)
% Funktion zur Überlagerung des Signals mit normalverteilten Rauschen
% Eingabeparameter: Eingangssignal (x), sowie dem gewünschten Signal-Rauschabstand (snrdb);
% Ausgabeparameter (y): Eingangssignal mit additiven Rauschen

    M = size(x);
    
    Eb = mean(abs(x).^2);
    %Eb = x/sqrt(2);    % Singalleistung
    
    if (snrlin> 0) 
        N0 = Eb./snrlin;    % Rauschleistung
    else
        N0 = Eb;
    end
    
    sigma = sqrt(N0/2); % Normierungsfaktor
    
    noise= Eb.* (sigma.* randn(M)+ 1j * sigma.* randn(M)); 

    y = x + noise; % additives Kanalrauschen durch senden ueber einen Kanal
end