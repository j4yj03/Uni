function y=radioFadingChannel(i, nSamp, K, Nr)
% Funktion zu Bestimmung der Kanalkoeffizienten
% Eingabeparameter: die Laufvariable i, sowie die Anzahl der gewünschten
% Kanalkoeffizienten (nSamp), als auch der K-Parameter (K) und auch die Anzahl
% der Emfangsantennen (Nr)
% Ausgabeparameter: Es wird eine Matrix mit den normalverteilten Kanalkoeffizienten
% ausgegeben
    %H_NLOS = zeros(Nr,nSamp); % Speicher fuer den Matrix reservieren (wird mit 0en initialisiert)
    %H_LOS = zeros(Nr,nSamp);

    mean2 = K/(K+1);   % Leistung der LOS Komponente 
    sigma2 = 1/(K+1);  % Leistung der NLOS Komponent
    omega = 1/sqrt(2);    % Skalierungsfaktor (mittlere Leistung des Signals)

    H_NLOS = sqrt(sigma2) * (omega * (randn(Nr,nSamp) + 1j*randn(Nr,nSamp))); % normalisierte h-Koeffizienten NLOS
    H_LOS = sqrt(mean2) * (ones(Nr,nSamp));                                  % h-Koefizient LOS
    
    y = (H_LOS + H_NLOS); % normal verteilte Kanalkoefizienten
    
    if (i==1)   % Nur die ersten Kanalkoeffizienten sollen geplottet werden
        plotCoeff(y(1,:),K)
    end
end