% evaluateRecognitionSystem.m
% This script evaluates nearest neighbour recognition system on test images
% load traintest.mat and classify each of the test_imagenames files.
% Report both accuracy and confusion matrix

% load training and testing features
clear
% load 'visionHarris.mat'
load 'visionRandom.mat'
distance_method = 'chsq'; % 'chsq'


% find NN and confusion matrix
clsNum = max(trainLabels);
confusion_matrix = zeros(clsNum, clsNum);

load '../data/traintest.mat'
for i = 1 : numel(test_labels)
   load(['../data/' strrep(test_imagenames{i},'.jpg','Random.mat')]); % 'Random.mat' for random
   testFeat = getImageFeatures(wordMap, 100);
   dist = getImageDistance(testFeat, trainFeatures, distance_method);
   [~, idx] = sort(dist', 'ascend');
   label_pred = trainLabels(idx(1), 1);
   label_gt = test_labels(1, i);
   confusion_matrix(label_pred, label_gt) = confusion_matrix(label_pred, label_gt) + 1;
end
% disp(confusion_matrix)

% acc
acc = trace(confusion_matrix) / numel(test_labels);
disp(acc)