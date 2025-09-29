clear
clc
clear helperModClassGetModulator
clear helperModClassGetSource
%% Init
DataJson = struct();
DataJson.dataset_name = "sig_rayleigh";
data_struct = struct('id', {}, 'image_path', {}, 'label', {});

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
dataDirectory = fullfile("Sig_rayleigh"); %默认保存路径
disp("Data file directory is " + dataDirectory)
%% Generate
i = 1;
for snrLevel = 1:numSNR_level                           %one cycle SNR
    % channel
    rayleigh_channel = comm.RayleighChannel( ...
        'SampleRate', fs, ...
        'PathDelays', [0 1.8 2.5] / fs, ...
        'AveragePathGains', [0 -2 -4], ...
        'MaximumDopplerShift', 4);
    channelInfo = info(rayleigh_channel);
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
            outChannel = rayleigh_channel(y);
            rxSamples = awgn(outChannel,label_snr,'measured');
            frame(:,p) = helperModClassFrameGenerator(rxSamples, spf, spf, transDelay, sps);
        end
        data(:,1,:) = real(frame.');
        data(:,2,:) = imag(frame.');
        fileName = fullfile(dataDirectory,char(modulationTypes(modType)),sprintf("%s%s%ddB%03d",modulationTypes(modType),"_",SNR(snrLevel)));
        data_struct(i).id = i;
        data_struct(i).data_path = (fileName+'.mat');
        data_struct(i).label = char(modulationTypes(modType));
        save(fileName,"data")
        i = i+1;
    end
end

DataJson.data = data_struct;
json_string = jsonencode(DataJson);
file_path = 'sig_rayleigh.json';
fid = fopen(file_path, 'w');
fprintf(fid, '%s', json_string);
fclose(fid);
