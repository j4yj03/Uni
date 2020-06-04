% Radio communication system simulation script
% =========================================================================
% Author: Sidney Goehler (544131) HTW Berlin WiSe 18/19
% Digitale Funksysteme - Prof. Dr.-Ing. Markus Noelle
% =========================================================================

clc; 
clear variables; % clear all variables
%addpath('subfunctions'); % add directory "subfunctions" to path

%% global simulation parameters
ebN0dB = 0:30; % SNR (per bit) in dB
K=1;        % Rice K-Faktor (P_LOS / P_NLOS)

nMinErr=100;
nBitsPerLoop =50e3; % simulate nBits bits per simulation loop
nMaxBits= 100*nBitsPerLoop;
constellation = [-1-1j, 1-1j, -1+1j, 1+1j]; % constellation of the modulation format here: QPSK with Gray mapping
%constellation = [-3-3j,-1-3j,1-3j,3-3j,-3-1j,-1-1j,1-1j,3-1j,-3+1j,-1+1j,1+1j,3+1j,-3+3j,-1+3j,1+3j,3+3j];  %constellation for 16QAM
%constellation = [exp(0),exp(j*pi/4),exp(j*3*pi/4),exp(j*pi/2),exp(j*7*pi/4),exp(j*3*pi/2),exp(j*pi),exp(j*5*pi/4)];  %constellation for 8-PSK

%%=========================================================================
power = rms(constellation);

ebN0lin = 10.^(ebN0dB/10);   % Lineare SNR; wird fuer die Berechnung der theoretischen BER benoetigt
esN0lin = ebN0lin * log2(length(constellation));    % SNR pro Symbol (linear)
esN0dB = 10*log10(esN0lin);		% SNR pro Symbol in dB


%theoretische Fehlerraten
P_AWGN_QPSK = 0.5*(erfc(sqrt(ebN0lin)));
P_AWGN_8PSK= 0.5*(erfc(sqrt((3*sin(pi/8)^2).*ebN0lin)));
P_AWGN_16QAM = 0.5*(erfc(sqrt((2/5)* ebN0lin)));
P_AWGN_64QAM = 0.5*(erfc(sqrt((1/7)*ebN0lin)));
P_RAY_QPSK = 0.5.*(1-(sqrt(ebN0lin./(ebN0lin+1))));
%Ricekanal Fehlerfunktion
f = @(x) (((1+K)*sin(x).^2)./((1+K)*(sin(x).^2)+ebN0lin)).*exp(-(K*ebN0lin./((1+K)*sin(x).^2+ebN0lin)));
P_RICE_QPSK =(1/pi).*integral(f,0,(pi/2),'ArrayValued', true);

% Initialisierung der Vektoren und Variablen
ERate=[];
durchRate=zeros(1,length(esN0dB));
anzFehler=0;i=1;bits=zeros(1,nBitsPerLoop);


% ENDE global simulation parameters
% =========================================================================
%% simulations loop...
for i=1:length(esN0dB)
    tic
    totalFehler=0;
    nProcessedBits=0;
    j=1;

    sprintf('Runde %d: %.4fdB..', i,esN0dB(i))  % Ausgabe der aktuellen SNR    
    
    while(totalFehler < nMinErr && nProcessedBits < nMaxBits)
        
        % Erzeugung von Bits und Modulierten Bits
        bits = generateBits(nBitsPerLoop);
        mappedSymbols = bitMapper(bits,constellation);

        % Kanal: Symbole mit Kanalkoeffizient kompensiert
        compensatedSymbols=fadingChannel(i,mappedSymbols,esN0dB(i),K);
		
		% Kanalkoeffizienten sind bekannt, das der Kanal ideal geschaetz wurde
		% entschiedene Symbole mit mittlerer Leistung Skaliert
        decidedSymbols=decision(compensatedSymbols.*power,constellation);
        
        demappedBits=demapper(decidedSymbols,constellation);
        
        % Fehlerraten ermitteln und zwischenspeichern
        [anzFehler, fehlerRate] = countErrors(demappedBits,bits);
        totalFehler=totalFehler+anzFehler;
        nProcessedBits=nProcessedBits+nBitsPerLoop;
        ERate(j)=fehlerRate;
        j=j+1;
    end
    durchRate(i) = mean(ERate); % durchschnittliche Fehlerrate wird ermittelt
    sprintf('Fehlerrate: %.10f%%..', durchRate(i)*100)  % Ausgabe der Fehlerrate
    clear ERate;    % aktuelle Fehlerrate wird geloescht
    toc
end

% Ergebnisse plotten
figure('Name','Werte');
semilogy(ebN0dB,P_AWGN_QPSK,'-r',ebN0dB,P_RICE_QPSK,'-c',ebN0dB,durchRate,'r.');
legend('AWGN theoretisch','Rice theoretisch','berechnet');
hold on;
grid on;
grid minor;
%setzt die Grenzen der y-Achse & x-Achse
ylim([10^(-7) 10^0]);
xlim([0 30]);
    
%% ========================================================================
% =========================================================================
% Funktionen
% =========================================================================
% =========================================================================
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
	
    transmittedSymbols = mappedSymbols.*h;         % Symbole werden ueber den Kanal "gesendet"
	
    receivedsymbols=awgn(transmittedSymbols,SNR,'measured');  % additives Kanalrauschen durch senden ueber einen Kanal
	
    y = receivedsymbols./h;       % Symbole werden kompensiert/entzerrt. h(x) ist bekannt, das der Kanal ideal geschaetzt wurde
end
%==========================================================================
function y=radioFadingChannel(i, nSamp, K, Nr)
% Funktion zu Bestimmung der Kanalkoeffizienten
% Eingabeparameter: die Laufvariable i, sowie die Anzahl der gewünschten
% Kanalkoeffizienten (nSamp), als auch der K-Parameter (K) und auch die Anzahl
% der Emfangsantennen (Nr)
% Ausgabeparameter: Es wird eine Matrix mit den normalverteilten Kanalkoeffizienten
% ausgegeben
	for Nr
		mean2 = K/(K+1);   % Leistung der LOS Komponente 
		sigma2 = 1/(K+1);  % Leistung der NLOS Komponent
		omega = 1/sqrt(2);    % Skalierungsfaktor

		H_NLOS = sqrt(sigma2) * (omega * (randn(1,nSamp) + 1j*randn(1,nSamp))); % normalisierte h-Koeffizienten NLOS
		H_LOS = sqrt(mean2) * (ones(1,nSamp));                                  % h-Koefizient LOS
		
		y[Nr] = (H_LOS + H_NLOS); % normal verteilte Kanalkoefizienten
		
		if (i==1)   % Nur die ersten Kanalkoeffizienten sollen geplottet werden
			plotCoeff(y[Nr],K)
		end
	end
   
end
%=======================================================================================================================
function plotCoeff(coeff,K)
% Eingabeparameter: der Vektor mit den Koeffizienten (coeff), sowie der 
% K-Parameter(K); 
% Ausgabeparameter: keine (es werden nur plots erzeugt)
    
    sigma2 = 1/(K+1); % Leistung der NLOS Komponent
    figure('Name','Amplitudeverteilung der Kanalkoeffizienten');
    x1=sqrt(real(coeff).^2 + imag(coeff).^2); %Amplitude der Koeff
    x2=linspace(0,3.5,1000);
    histogram(x1,'Normalization','pdf'); % plotten der Amplituden
    % Plot der theoretischen PDF
    pdf=((x2./ (sigma2./2)).*exp(-((x2./(sqrt(2)*(sqrt(sigma2./2)))).^2 + K)).* besseli(0,(( x2 .* sqrt(2))./sqrt(sigma2./2)).* sqrt(K)));
    hold on;
    grid on;
    plot(x2,pdf,'r')
    legend('berechnet','theoretisch')
    % plotten der Phase
    figure('Name','Phase der Kanalkoeffizienten');
    x1=atan2(imag(coeff),real(coeff)); %Phase der Koeff
    histogram(x1,'Normalization','pdf');
    grid on;
    legend('berechnet')
end
%==========================================================================
function y = generateBits(x) 
% Funktion zur gleichverteilten Erzeugung von Bits
% Eingabeparameter: die Anzahl der zu erzeugenden Bits (x);
% Ausgabeparameter (y): Anzahl der erzeugten Bits

    if (x <= 0) || ( x >= 2e15) 
        error('groesse des Vektors darf nicht kleiner als 0 und nicht groesser als 2^15');
    end
    
    y = randi([0 1], 1,x);    % Bit-Vektor erzeugen

end
 %=========================================================================
function y = bitMapper(bits, constellation)
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
 %======================================================================================================================
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
 %=========================================================================
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
 %=========================================================================
function [nErr, ber] = countErrors(receivedBits, sendBits)
% Funktion zur Ermittlung der Anzahl der Fehler, als auch der Bitfehlerrate
% Eingabeparameter: die Empfangenen Bits (received Bits), sowie die
% gesendeten Bits (sendBits)
% Ausgabeparameter: die Anzahl der Fehler (nErr), sowie die Fehlerrate
% (ber) werden ausgegeben
    nErr = length(find(receivedBits ~= sendBits)); % laenge des Ergebnisvektors entspricht der Anzahl der fehlerhaften Bits
    ber = nErr/length(sendBits);
  
end