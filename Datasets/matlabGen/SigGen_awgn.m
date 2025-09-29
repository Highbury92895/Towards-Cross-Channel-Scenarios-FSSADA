clear
clc

clear helperModClassGetModulator
clear helperModClassGetSource
%% Init
% generate type and snr
modulationTypes = categorical(["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK"]);
SNR  = -6:2:18;
numModulationTypes = length(modulationTypes);       
numSNR_level = length(SNR);  

numFramesPerModType = 1000;             %每个信号产生帧数

sps = 8;                  % Samples per symbol
spf = 1024;                % Samples per frame
symbolsPerFrame = spf / sps; % Symbol per frame

fs = 200e3;             % Sample rate
transDelay = 50;
 
dataDirectory = fullfile('E:\','AMC','AMC322',"DateSet322",'Sig_awgn'); %默认保存路径
disp("Data file directory is " + dataDirectory)

%% Generate
for snrLevel = 1:numSNR_level                            %one cycle SNR
    for modType = 1:numModulationTypes                   %two cycle modtype
      fprintf('Generating %s frames %d SNR(dB)\n',modulationTypes(modType),SNR(snrLevel))
    
      label= char(modulationTypes(modType));
      label_snr = SNR(snrLevel);
      dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
      modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs);
      frame = zeros(spf,numFramesPerModType);
      for p=1:numFramesPerModType                       %three cycle n_frame
        x = dataSrc();
        y = modulator(x);
        rxSamples = awgn(y,label_snr,'measured');
        frame(:,p) = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
      end
      data(:,1,:) = real(frame.');
      data(:,2,:) = imag(frame.');
      fileName = fullfile(dataDirectory,sprintf("%s%s%ddB%03d",modulationTypes(modType),"_",SNR(snrLevel)));
      save(fileName,"data","label","label_snr") 
     end
end
% else
%   disp("Data files exist. Skip data generation.")
% end