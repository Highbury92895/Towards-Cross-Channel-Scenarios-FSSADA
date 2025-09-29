clear
clc
clear helperModClassGetModulator
clear helperModClassGetSource
%% Init


% generate type and snr
modulationTypes = categorical(["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK","B-FM","DSB-AM","SSB-AM"]);
SNR  = -6:2:18;
numModulationTypes = length(modulationTypes);
numSNR_level = length(SNR);

numFramesPerModType = 1000;           %每个信号产生帧数

sps = 8;                   % Samples per symbol
spf = 1024;                % Samples per frame
symbolsPerFrame = spf / sps; % Symbol per frame

fs = 200e3;             % Sample rate
transDelay = 50;
%dataDirectory = fullfile("C:\Users\admin\Desktop\AI-aided\projects\matlab\MatlabGen322","Sig_rayleigh"); %默认保存路径
dataDirectory = fullfile("Sig_rician"); %默认保存路径
disp("Data file directory is " + dataDirectory)

DataJson = struct();
DataJson.dataset_name = "sig_rician";
DataJson.numFrame = numFramesPerModType;
DataJson.numSample = spf;
data_struct = struct('id', {}, 'data_path', {}, 'label', {},'snr', {});
%% Generate
i = 1;
for snrLevel = 1:numSNR_level                           %one cycle SNR
    % channel
    rician_channel = comm.RicianChannel(...
        'SampleRate', fs, ...
        'PathDelays', [0 4] / fs, ...
        'AveragePathGains', [0 -3], ...
        'MaximumDopplerShift', 10);
    channelInfo = info(rician_channel);
    for modType = 1:numModulationTypes                   %two cycle modtype
        fprintf('Generating %s frames %d SNR(dB)\n',modulationTypes(modType),SNR(snrLevel))
        label= char(modulationTypes(modType));
        label_snr = SNR(snrLevel);
        dataSrc = helperModClassGetSource(modulationTypes(modType), sps, 2*spf, fs);
        modulator = helperModClassGetModulator(modulationTypes(modType), sps, fs);
        frame = zeros(spf,numFramesPerModType);
        for p=1:numFramesPerModType                       %three cycle n_frame
            % Channel
            x = dataSrc();
            y = modulator(x);
            outChannel = rician_channel(y);
            rxSamples = awgn(outChannel,label_snr,'measured');
            frame(:,p) = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
        end
        data(:,1,:) = real(frame.');
        data(:,2,:) = imag(frame.');
        fileName = fullfile(dataDirectory,char(modulationTypes(modType)),sprintf("%s%s%ddB%03d",modulationTypes(modType),"_",SNR(snrLevel)));
        data_struct(i).id = i;
        data_struct(i).data_path = (fileName+'.mat');
        data_struct(i).label = char(modulationTypes(modType));
        data_struct(i).snr = label_snr;
        save(fileName,"data")
        i = i+1;
    end
end

DataJson.data = data_struct;
json_string = jsonencode(DataJson);
file_path = 'sig_rician.json';
fid = fopen(file_path, 'w');
fprintf(fid, '%s', json_string);
fclose(fid);
