function y=antennaCombining(x, h, combMethod)
% Funktion zur Bestimmung der Linearkombination der einzelnen
% Empfangssignale
% Eingabeparameter: die Antennensignale (x) und die Kanalkoeffizienten (h)
% als Matrix, sowie die Kombinationsmethode (combMethod)
% Ausgabeparameter: Der Vektor der kombinierten Singale (y)

    if strcmp(combMethod,'mrc')
    elseif strcmp(combMethod,'egc')
    elseif strcmp(combMethod,'sdc')
    else
        error(['Unknown combination method: ' convmtrx]);
    end
    y=0;
end