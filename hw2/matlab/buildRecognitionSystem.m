% buildRecognitionSystem.m
% This script loads the visual word dictionary (in dictionaryRandom.mat or dictionaryHarris.mat) and processes
% the training images so as to build the recognition system. The result is
% stored in visionRandom.mat and visionHarris.mat.

% load and save .mat
clear
K = 100;
method = {'Random','Harris'};

for m = method
   load('../data/traintest.mat');
   dict_path = ['dictionary', m{1,1}, '.mat'];
   load(dict_path);
   trainFeatures = zeros(numel(train_imagenames), K);
   trainLabels = train_labels';
   for i = 1 : numel(train_imagenames)
      I = imread(['../data/', train_imagenames{1,i}]);
      wordMap = getVisualWords(I, dictionary, filterBank);
      trainFeatures(i,:) = getImageFeatures(wordMap, size(dictionary, 1));
      fprintf([num2str(i), '\n']);
   end
   if m{1,1} == 'Harris'
       save visionHarris.mat dictionary filterBank trainFeatures trainLabels
   else
       save visionRandom.mat dictionary filterBank trainFeatures trainLabels
   end
end

