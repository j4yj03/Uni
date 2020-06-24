function y=setSNR2(x, snrdb)
%Funktion SNR
    %Px = rms(x)^2;
    %Pn = snrdb * Px;
    %y = x + Pn.* randn(size(x))
    y = awgn(x,snrdb,'measured');  % additives Kanalrauschen durch senden ueber einen Kanal
end