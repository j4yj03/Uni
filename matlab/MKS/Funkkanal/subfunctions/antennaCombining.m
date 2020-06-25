function y=antennaCombining(x, h, combMethod)
% Funktion zur Bestimmung der Linearkombination der einzelnen
% Empfangssignale
% Eingabeparameter: die Antennensignale (x) und die Kanalkoeffizienten (h)
% als Matrix, sowie die Kombinationsmethode (combMethod)
% Ausgabeparameter: Der Vektor der kombinierten Singale (y)

    if strcmp(combMethod,'sum')
    elseif strcmp(combMethod,'MRC')
    elseif strcmp(combMethod,'EGC')
    elseif strcmp(combMethod,'SDC')
        %selection diversity r
        %kanalkoeffizienten mit der größten amplitude
    else
        error(['Unknown combination method: ' convmtrx]);
    end
    y=0;
end