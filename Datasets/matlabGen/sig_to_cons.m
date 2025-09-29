
modulationTypes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK"];
SNR  = -6:2:18;
for i = 1:8
    for j = 1:13
    mod = modulationTypes(i);
    snr = SNR(j);
    dataiq = load('E:\AMC\AMC322\DateSet322\Sig_rayleigh\'+mod+'_'+snr+'dB.mat').data; 

    data = IQtoCons(dataiq);
    fileName = fullfile('E:\AMC\AMC322\DateSet322\','Sig_rayleigh1_cons\',sprintf("%s%s%ddB%03d",char(mod),"_",snr));
    save(fileName,"data","mod","snr") 
    end
end