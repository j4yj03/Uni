function y=fadingChannel(i,mappedSymbols, SNR, K, Nr)
% Funktion zu Simulation des Funkkanals
% Eingabeparameter: da die Koeffizienten nur einmal geplottet werden sollen
% wird die Laufvariable i benötigt. Die gemappten Symbole (mappedSymbols),
% der gewünschte Signal-Rauschabstand (SNR) sowie der gewünschte
% K-Parameter (K);
% Ausgabeparameter: es werden die kompensierten Symbole ausgegeben (y)
    if (~exist('K','var'))      % wenn kein K angegeben ist wird ein Reyleighkanal simuliert
        K=0;
    end
    
    nSamp = length(mappedSymbols);
    
    h = radioFadingChannel(i, nSamp, K, Nr); % Kanalkoeffizienten generieren
	
    transmittedSymbols = mappedSymbols.*h;         % Symbole werden ueber den Kanal "gesendet"
	
    %receivedsymbols = setSNR(transmittedSymbols,SNR);  % additives Kanalrauschen durch senden ueber einen Kanal
	%receivedsymbols = awgn(transmittedSymbols,SNR,'measured');  % additives Kanalrauschen durch 
    
    y = receivedsymbols./h;       % Symbole werden kompensiert/entzerrt. h(x) ist bekannt, das der Kanal ideal geschaetzt wurde

    %y = antennaCombining(receivedsymbols, h, 'sdc');
end