clc; 
clear variables; % clear all variables
addpath('subfunctions'); % add directory "subfunctions" to path

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

power = sqrt(mean(constellation.^2));

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
        mappedSymbols = mapper(bits,constellation);

        % Kanal
        compensatedSymbols=fadingChannel(i,mappedSymbols,esN0dB(i),K);

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