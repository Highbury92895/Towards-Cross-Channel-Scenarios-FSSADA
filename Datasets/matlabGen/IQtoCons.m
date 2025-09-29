function w = IQtoCons(data)
% data 1,2,1024
cons_size = 258;
cons1 = zeros(cons_size,cons_size);
num_sig = size(data,1);
for j = 1:num_sig
    count = 0;
    for i = 1:1024
        xi = data(j,1,i);
        xq = data(j,2,i);
        pos1 = floor((xi+2)*cons_size/4);
        pos2 = floor((xq+2)*cons_size/4);
        if (1 <= pos1 && pos1<= cons_size) &&  (1 <= pos2 && pos2<= cons_size)
                cons1(pos1,pos2) = cons1(pos1,pos2)+1;
                count = count+1;
        else
            continue;
        end
    end
    W = 32;
    H = 32;
    % cons2 = zeros(227,227);
    filter = ones(H,W);
    w(j,:,:) = conv2(cons1,filter,'valid');
    %imagesc(w)
end



