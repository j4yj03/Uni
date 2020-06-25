N_frame=100;
N_packet=100000;
NR=3; %  оличество разнесенных антенн
SNRdBs=[0:1:10];
for i_SNR=1:length(SNRdBs)
    SNRdB=SNRdBs(i_SNR);
    sigma=sqrt(0.5/(10^(SNRdB/10)));
    for i_packet=1:N_packet
      %генерирование сообщени€
      msg_symbol=randint(N_frame,1);
      % BPSK modulation
      sym_tab=exp(j*[-pi 0]);
      X = (sym_tab(msg_symbol+1))';
      %»cкажение сигнала и прием его на две антенны
      H = (randn(N_frame,NR)+j*randn(N_frame,NR));%мультипликативна€ помеха
      Z = (sigma*(randn(N_frame,NR)+j*randn(N_frame,NR)));%адитивна€ помеха         
      Ysum=0;
    for i_an=1:NR
      Y(:,i_an) = H(:,i_an).*X(:) + Z(:,i_an);     %прин€тые сигналы на 1 и 2ой антенне c помехой
      W_mrc=conj(H(:,i_an));                       %вычисление весового коэфициента
      Ysum = Ysum + Y(:,i_an).*W_mrc;              %комбинирование взвешанных сигналов
      end
 sigDemod = (abs(angle(Ysum))<pi/2);
   er=0; 
 for i = 1:N_frame
     if msg_symbol(i,1)~=sigDemod(i,1)
         er=er+1;
     end
 end
 BERp(i_packet)=  er/N_frame;
       end 
     BERmrc(i_SNR) = sum(BERp)/N_packet;
 end;
% semilogy(SNRdBs,BER), grid on, axis([SNRdBs([1 end]) 1e-6 1e0])