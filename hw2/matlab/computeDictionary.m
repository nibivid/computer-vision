clear

method = {'Harris', 'Random'};
load '../data/traintest.mat';
alpha = 50;
K = 100;
for m = method
    dictionary = getDictionary(train_imagenames, alpha, K, m{1,1});
    filterBank = createFilterBank();
    name = ['dictionary' m '.mat'];
    save name filterBank dictionary
end