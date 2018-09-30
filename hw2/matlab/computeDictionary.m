clear

method = {'Harris','Random'};
load '../data/traintest.mat';
alpha = 50;
K = 100;
for m = method
    dictionary = getDictionary(train_imagenames, alpha, K, m{1,1});
    filterBank = createFilterBank();
    if m{1,1} == 'Harris'
        save dictionaryHarris.mat filterBank dictionary
    else
        save dictionaryRandom.mat filterBank dictionary
    end
end
