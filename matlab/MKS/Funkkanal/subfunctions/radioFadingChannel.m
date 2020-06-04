function y=radioFadingChannel(i,nSamp,K)
% Funktion zu Bestimmung der Kanalkoeffizienten
% Eingabeparameter: die Laufvariable i, sowie die Anzahl der gewünschten
% Kanalkoeffizienten (nSamp), als auch der K-Parameter (K)
% Ausgabeparameter: Es werden die normalverteilten Kanalkoeffizienten
% ausgegeben
    mean2 = K/(K+1);   % Leistung der LOS Komponente 
    sigma2 = 1/(K+1);  % Leistung der NLOS Komponent
    omega = 1/sqrt(2);    % Skalierungsfaktor

    H_NLOS = sqrt(sigma2) * (omega * (randn(1,nSamp) + 1j*randn(1,nSamp))); % normalisierte h-Koeffizienten NLOS
    H_LOS = sqrt(mean2) * (ones(1,nSamp));                                  % h-Koefizient LOS
    
    y = (H_LOS + H_NLOS); % normal verteilte Kanalkoefizienten
    
    if (i==1)   % Nur die ersten Kanalkoeffizienten sollen geplottet werden
        % plotCoeff(y,K)
    end
   
end