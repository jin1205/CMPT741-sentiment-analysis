%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% author: Jin Zhang
% date: date for release this code

clear; clc;
%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()

% read data
[data, wordMap] = read_data();

% separate data into training_data and validation_data
training_data = data(1:length(data)*0.8,:);
validation_data = data(length(data)*0.8+1: end,:);

% pad data to make sure the sentence is longer than the filter size
wordMap('<PAD>') = length(wordMap) + 1;
wordMap('<UNK>') = length(wordMap) + 1;

% save wordMap to 
% fileID = fopen('vocab.dat','w');
% formatSpec = '%d %s\n';
% vocab = wordMap.keys();
% [nrows,ncols] = size(vocab);
% for col = 1:ncols
%     fprintf(fileID,formatSpec,wordMap(vocab{col}), vocab{col});
% end
% fclose(fileID);

% init embedding
d = 300;
total_words = length(wordMap);

% random sample from normal distribution
% with mean = 0, variance = 0.1

% T = normrnd(0, 0.1, [total_words, d]);
T = importdata('wordvector_300_840B.txt');
% init filters 
filter_size = [2, 3, 4];
n_filter = 2;

W_conv = cell(length(filter_size), 1);
B_conv = cell(length(filter_size), 1);

for i = 1: length(filter_size)
    % get filter size
    f = filter_size(i);
    % init W with: FW x FH x FC x K
    W_conv{i} = normrnd(0, 0.1, [f, d, 1, n_filter]);
    B_conv{i} = zeros(n_filter, 1);
end

% init output layer
total_filters = length(filter_size) * n_filter;
n_class = 2;
W_out = normrnd(0, 0.1, [total_filters, n_class]);
B_out = zeros(n_class, 1);

% init gradient descent parameters
numIterations = 10;
rate = 0.008;

%% Section 2: training

% for each example in train.txt do
loss = cell(1, numIterations);
for t = 1: numIterations
    loss_t = 0;
    for i = 1: length(training_data)        
        sentence = training_data{i, 2};
        if length(sentence) < 4
            for k = length(sentence)+1: 4
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
        
        % section 2.1 forward propagation and compute the loss
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
        z = vl_nnconcat(pool_res, 3);

        % compute loss
        % o: value of output layer
        % y: ground truth label (1 or 2)
        if training_data{i, 3} == 1
            y = 1;
        else y = 2;
        end
        z_reshape = reshape(z, length(filter_size) * n_filter, 1);
        o = vl_nnconv(z_reshape, reshape(W_out, length(filter_size) * n_filter, 1, 1, 2), B_out);
        loss_t = loss_t + vl_nnloss(reshape(o, 1, 1, 2, 1), y);  
        
        % section 2.2 backward propagation and compute the derivatives
        dlossdo = vl_nnloss(reshape(o, 1, 1, 2, 1), y, 1);
        [dlossdz, dlossdW_out, dlossdB_out] = ...
            vl_nnconv(z_reshape, reshape(W_out, length(filter_size) * n_filter, 1, 1, 2), B_out,dlossdo);
        
        dlossdpool_res = vl_nnconcat(pool_res, length(filter_size), ... 
            reshape(dlossdz, 1, 1, length(filter_size) * n_filter));
        
        dlossdrelu = cell(1, length(filter_size));
        dlossdconv = cell(1, length(filter_size));
        dlossdX = cell(1, length(filter_size));
        dlossdW_conv = cell(1, length(filter_size));
        dlossdB_conv = cell(1, length(filter_size));
        
        for k = 1: length(filter_size)
            conv = cache{1, k};
            relu = cache{2, k};
            sizes = size(conv);
            dlossdrelu{k} = vl_nnpool(relu, [sizes(1), 1], dlossdpool_res{k});
            dlossdconv{k} = vl_nnrelu(conv, dlossdrelu{k});
            [dlossdX{k}, dlossdW_conv{k}, dlossdB_conv{k}] = ...
            vl_nnconv(X, W_conv{k}, B_conv{k},dlossdconv{k});
        end
        
        % section 2.3 update the parameters
        W_out = W_out - rate .* reshape(dlossdW_out, 6, 2);
        B_out = B_out - rate .* dlossdB_out;
        for k = 1: length(filter_size)
            W_conv{k} = W_conv{k} - rate .* dlossdW_conv{k};
            B_conv{k} = B_conv{k} - rate .* dlossdB_conv{k};
            X = X - rate .* dlossdX{k};            
        end
        for k = 1:length(sentence)
            index = word_indexs(k);
            T(index, :) = X(k,:);
        end
    end
    
    % print loss for each epoch   
    fprintf('Epoch %d: loss: %.2d, ', t, loss_t);
    
    loss{t} = loss_t;
    
    % calculate accuracy on training data
    error_train = 0;
    for i = 1: length(training_data)
        sentence = training_data{i, 2};
        if length(sentence) < 4
            for k = length(sentence)+1: 4
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
        o = reshape(z_reshape * W_out, 2, 1) + B_out;
        if o(1) > o(2)
            y = 1;
        else y = 0;
        end
        error_train = error_train + abs(y - training_data{i, 3});
    end
    
    
    % calculate accuracy on validation data
    error_val = 0;
    for i = 1: length(validation_data)
        sentence = validation_data{i, 2};
        if length(sentence) < 4
            for k = length(sentence)+1: 4
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
        o = reshape(z_reshape * W_out, 2, 1) + B_out;
        if o(1) > o(2)
            y = 1;
        else y = 0;
        end
        error_val = error_val + abs(y - validation_data{i, 3});
    end
    auc_train = 1 - error_train/length(training_data); 
    auc_val = 1 - error_val/length(validation_data);
    fprintf('training AUC: %f, validation AUC: %f,\n', auc_train, auc_val);
end

x = 1:t;
double_loss = cell2mat(loss);
plot(x, double_loss);



