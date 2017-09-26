


function allHists = wholeHistInfo(imageMatrix, base, nbins)

allHists = struct([]);
[nRow, nCol] = size(imageMatrix);

if base == 2
    noIterations = log2(min(nRow,nCol));
else
    noIterations = log(min(nRow,nCol))/log(2);
end
noIterations = floor(noIterations);

allHists(noIterations).Name = 'Whatever';
allHists(noIterations).histos = 1;
allTiles = MultiPartitioning(imageMatrix, base);

nLayers = size(allTiles, 2);

%{
% 200 for count = 1 : noIterations
    
    %200 nRowBlocks = base^count;
    %200 nColBlocks = base^count;
    %200 borders = allTiles(count).border    
    %200 allHists(count).Name = strcat('numberofRowBlocks', num2str(count));
    %200 allHists(count).histos = tiledHist( imageMatrix, borders, nbins, cRange);
    % allHists(count).histos = tiledHist(imageMatrix, nRowBlocks, nColBlocks,
    
    %200 % 100 struct1.histos = tiledImageHist(imageMatrix, nRowBlocks, nColBlocks, nbins, cRange);
    %200 % 100 struct1.Name = strcat('numberofRowBlocks',num2str(count^2));
    %200 % 100 allHists = [allHists, struct1]; 
%200 end
%}

for count = 1 : nLayers
    borders = allTiles(count).border;
    tilesHist = tiledHist(imageMatrix, borders, nbins);
    allHists(count).Name = strcat('Layer', num2str(count));
    allHists(count).histos = tilesHist;
end
end