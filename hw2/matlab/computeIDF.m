clear
load '../data/traintest.mat'

trainFeat = zeros(numel(train_labels), 100);

% get all train features
for i = 1 : numel(train_labels)
   load(['../data/' strrep(train_imagenames{i},'.jpg','.mat')]); % 'Random.mat' for random
   trainFeat(i,:) = getImageFeatures(wordMap, 100);
end

% compute IDF
IDF = zeros(1, 100);
for i = 1 : 100
    IDF(1,i) = numel(find(trainFeat(:,i))) + 1;     % add 1 to avoid divide by 0
end
IDF = log(numel(train_labels)./IDF);

save idf.mat IDF;