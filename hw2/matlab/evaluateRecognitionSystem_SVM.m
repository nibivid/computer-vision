% evaluateRecognitionSystem.m
% This script evaluates SVM recognition system on test images
% load traintest.mat and classify each of the test_imagenames files.
% Report both accuracy and confusion matrix
% code largely learn from MATLAB documentation: https://www.mathworks.com/help/stats/fitcsvm.html

clear
K = 100;

load('../data/traintest.mat');
load('../matlab/visionHarris.mat');

clsNames = unique(trainLabels);
clsNum = numel(clsNames);

% build a binary classifier for each class
% seems not working
% for i = 1 : clsNum
%    newLabels = (trainLabels == i);
%    SVMModels{i} = fitcsvm(trainFeatures, newLabels, 'ClassNames', [1,0],...
%                         'Standardize', true, 'KernelFunction', 'polynomial');
% end
for i = 1 : 2
    if i == 1
        t = templateSVM('KernelFunction','rbf');
    else
        t = templateSVM('KernelFunction','polynomial');
    end
    SVMModel = fitcecoc(trainFeatures,trainLabels,'Learners',t);
    if i == 1
        save visionSVM_rbf.mat clsNames clsNum SVMModel
    else
        save visionSVM_poly.mat clsNames clsNum SVMModel
    end
end

% classification and confusion matrix
testFeat = zeros(numel(test_labels), 100);
testScore = zeros(numel(test_labels), 8);
label_gt = test_labels';

% get all features
for i = 1 : numel(test_labels)
   load(['../data/' strrep(test_imagenames{i},'.jpg','.mat')]); % 'Random.mat' for random
   testFeat(i,:) = getImageFeatures(wordMap, 100);
end

% fit each SVM model for each class
% for i = 1 : clsNum
%     [label_pred, score] = predict(SVMModels{i}, testFeat);
%     testScore(:,i) = score(:,1);
% end
% % find max score cls for each example
% [~, maxScore] = max(testScore, [], 2);

for i = 1 : 2
    if i == 1
        load 'visionSVM_rbf.mat'
    else
        load 'visionSVM_poly.mat'
    end
    label_pred = predict(SVMModel, testFeat);
    confusion_matrix = confusionmat(label_gt, label_pred);
    acc = trace(confusion_matrix) / numel(test_labels);
    disp(confusion_matrix)
    disp(acc)
end
