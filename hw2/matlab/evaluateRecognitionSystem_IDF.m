clear
load 'visionHarris.mat'
% load 'visionRandom.mat'
load idf.mat


% find NN and confusion matrix
clsNum = max(trainLabels);
confusion_matrix = zeros(clsNum, clsNum);

load '../data/traintest.mat'
IDF_rep = repmat(IDF, size(trainFeatures,1),1);
trainFeatures = trainFeatures.*IDF_rep;

for i = 1 : numel(test_labels)
   load(['../data/' strrep(test_imagenames{i},'.jpg','.mat')]); % 'Random.mat' for random
   testFeat = getImageFeatures(wordMap, 100);
   testFeat = testFeat.*IDF;
   dist = getImageDistance(testFeat, trainFeatures, 'chsq');
   [~, idx] = sort(dist', 'ascend');
   label_pred = trainLabels(idx(1), 1);
   label_gt = test_labels(1, i);
   confusion_matrix(label_pred, label_gt) = confusion_matrix(label_pred, label_gt) + 1;
end
disp(confusion_matrix)

% acc
acc = trace(confusion_matrix) / numel(test_labels);
disp(acc)