clear
load '../data/traintest.mat'
method = {'HOG'};

trainNum = numel(train_imagenames);
testNum = numel(test_imagenames);
clsNum = numel(unique(train_labels));
trainLabels = train_labels';
testLabels = test_labels';

% extract features for training images
trainFeatures = [];
for i = 1 : trainNum
    img = imread(['../data/', train_imagenames{1,i}]);
    img = imresize(img, [64, 64]);      % to ensure the same size of features
    [feat, visualization] = extractHOGFeatures(img);
    trainFeatures = [trainFeatures; feat];
end
       
% extract features for testing images and find nearest neighbour
confusion_matrix = zeros(clsNum, clsNum);
for i = 1 : testNum
    img = imread(['../data/', test_imagenames{1,i}]);
    img = imresize(img, [64, 64]);      % to ensure the same size of features
    [feat, visualization] = extractHOGFeatures(img);
    dist = getImageDistance(feat, trainFeatures, 'chsq');
    [~, idx] = sort(dist', 'ascend');
    label_pred = trainLabels(idx(1), 1);
    label_gt = testLabels(i, 1);
    confusion_matrix(label_pred, label_gt) = confusion_matrix(label_pred, label_gt) + 1;
end

disp(confusion_matrix)
acc = trace(confusion_matrix) / numel(test_labels);
disp(acc)