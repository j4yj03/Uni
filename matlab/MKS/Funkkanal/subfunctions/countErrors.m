function [nErr, ber] = countErrors(receivedBits, sendBits)
% Funktion zur Ermittlung der Anzahl der Fehler, als auch der Bitfehlerrate
% Eingabeparameter: die Empfangenen Bits (received Bits), sowie die
% gesendeten Bits (sendBits)
% Ausgabeparameter: die Anzahl der Fehler (nErr), sowie die Fehlerrate
% (ber) werden ausgegeben
    nErr = length(find(receivedBits ~= sendBits)); % laenge des Ergebnisvektors entspricht der Anzahl der fehlerhaften Bits
    ber = nErr/length(sendBits);
  
end