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
    Mean = mean(abs(coeff).^2)
end