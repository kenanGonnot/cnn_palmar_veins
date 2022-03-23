clc;clear;
    Base = getBaseFiles('data/data_palm_vein/NIR');
    Test = getTestFiles('data/data_palm_vein/NIR');
      

    
    for i = 1:3000
        % Blue band %
        fprintf('%s \n', Base{i});
         train_x(:,:,i) = double(imread(Base{i}));
         test_x(:,:,i) = double(imread(Test{i}));
         trainName = strsplit(Base{i},'/');
         testName = strsplit(Test{i},'/');
         train_y(:,i) = lower(trainName{1,3} + '-' + trainName{1,4}(1:1));
         test_y(:,i) = lower(testName{1,3} + '-' + testName{1,4}(1:1));
    end

%end

train_x = train_x/255;
test_x = test_x/255;
%train_y = train_y;
%test_y = double(test_y);
%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;
opts.batchsize = 50;
opts.numepochs = 10;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

[er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
