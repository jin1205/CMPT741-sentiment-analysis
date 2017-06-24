% calculate accuracy on validation data
clear; clc;
load('parameter-345-93.mat');


% read in test data
headLine = true;
separater = '::';

data = cell(1000, 2);

fid = fopen('evaluation_set1.txt', 'r');
line = fgets(fid);

ind = 1;
while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    attrs = strsplit(line, separater);
    sid = str2double(attrs{1});
    
    s = attrs{2};
    w = strsplit(s);
   
    % save data
    data{ind, 1} = sid;
    data{ind, 2} = w;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
fprintf('finish loading test data\n');
fclose(fid);

filter_size = [3, 4, 5];
n_filter = 6;

fileID = fopen('test1.txt','w');
fprintf(fid, '%s::%s\n', 'id','label');
formatSpec = '%d::%d\n';
for i = 1: length(data)
    sentence = data{i, 2};
    if length(sentence) < 5
        for k = length(sentence)+1: 45
            sentence{k} = '<PAD>';   
        end
    end
    % get sentence matrix
    % words_indexs = [wordMap('i'), wordMap('like'),
    % ..., wordMap('!')]
    word_indexs = zeros(1, length(sentence));
    for j = 1: length(sentence)
        if (isKey(wordMap, sentence{j}) == 0)
            sentence{j} = '<UNK>';
        end
        word_indexs(j) = wordMap(sentence{j});
    end
    X = T(word_indexs, :);
        
    % forward propagation and compute the loss
    pool_res = cell(1, length(filter_size));
    cache = cell(2, length(filter_size));
    for k = 1: length(filter_size)
        % convolutional operation
        conv = vl_nnconv(X, W_conv{k}, B_conv{k});

        % apply activation function: relu
        relu = vl_nnrelu(conv);

        % 1-max pooling operation
        sizes = size(conv);
        pool = vl_nnpool(relu, [sizes(1), 1]);

        % important: keep these values for back-prop
        cache{2, k} = relu;
        cache{1, k} = conv;
        pool_res{k} = pool;
    end

    % concatenate
    z = vl_nnconcat(pool_res, length(filter_size));

    % compute loss
    % o: value of output layer
    % y: ground truth label (1 or 2)
    z_reshape = reshape(z, 1, length(filter_size) * n_filter);
    o = reshape(z_reshape * (W_out), 2, 1) + B_out;
    if o(1) > o(2)
        y = 1;
    else y = 0;
    end
    %save prediction to output 
    fprintf(fileID,formatSpec,i,y);
end
fclose(fileID);