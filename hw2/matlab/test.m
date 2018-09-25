clear

% part 1.1
filterBank = createFilterBank();

% part 1.2
% I = imread('../data/desert/sun_adpbjcrpyetqykvt.jpg');
% I = imread('../data/campus/sun_abslhphpiejdjmpz.jpg');
% I = imread('../data/campus/sun_adiqdyqsqarvtact.jpg');
% I = imread('../data/airport/sun_aewkrrhvwhkvbcix.jpg');
I = imread('../data/bedroom/sun_abelvucjanbnkioi.jpg');
%filterResponses = extractFilterResponses(I, filterBank);
% points = getHarrisPoints(I, 500, 0.05);
% points = getRandomPoints(I, 500);

% load dictionaryHarris.mat
load dictionaryRandom.mat
wordMap = getVisualWords(I, dictionary, filterBank);
