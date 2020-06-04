function y=fadingChannel(i,mappedSymbols,SNR,K)
% Funktion zu Simulation des Funkkanals
% Eingabeparameter: da die Koeffizienten nur einmal geplottet werden sollen
% wird die Laufvariable i benötigt. Die gemappten Symbole (mappedSymbols),
% der gewünschte Signal-Rauschabstand (SNR) sowie der gewünschte
% K-Parameter (K);
% Ausgabeparameter: es werden die kompensierten Symbole ausgegeben (y)
    if (~exist('K','var'))      % wenn kein K angegeben ist wird ein Reyleighkanal simuliert
        K=0;
    end
    h=radioFadingChannel(i,length(mappedSymbols),K); % Kanalkoeffizienten generieren
    transmittedSymbols=mappedSymbols.*h;         % Symbole werden ueber den Kanal "gesendet"
    receivedsymbols=awgn(transmittedSymbols,SNR,'measured');  % Kanalrauschen
    y=receivedsymbols./h;       % Symbole werden kompensiert
end