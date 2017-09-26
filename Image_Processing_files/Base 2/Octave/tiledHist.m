% This function read the image, devides it into
% smaller blocks and makes histogram for each block.

% nRowBlocks is the number of row blocks,
% NColBlocks is the number of column blocks
% nbins is the number of bins we want to put in the signature file.


% rowPartition is the borders on the image that creates different blocks
% horizontaly

% colPartition is the borders on the image that creates different blocks
% vertically

% nbins is the number of bins that we want to devide cRange by.

function tilesHist = tiledHist(imageMatrix, borders, nbins)
 % 100 signature = tiledImageHist(imageMatrix, nRowBlocks, nColBlocks, nbins, cRange)
 % 100 signature = zeros(nbins,nColBlocks,nRowBlocks);
 rowBorders = borders(:,1);
 colBorders = borders(:,2);

 tilesHist = zeros(nbins, length(colBorders), length(rowBorders));
%{
% [nRows, nCol] = size(imageMatrix);
%
% rowCenter and colCenter give the row and column number of center
% pixels.

% rowPart and colPart give the boundaries of Partitions.
%
% 100 [~, ~, rowPart, colPart] = MP(nRows, nCol, nRowBlocks, nColBlocks);
%}

 lenRowPart = length(rowBorders);
 lenColPart = length(colBorders);

 rowBorders = [0; rowBorders];
 colBorders = [0; colBorders];

 for rowCount = 1:lenRowPart
     for colmCount=1:lenColPart
         block = imageMatrix (rowBorders(rowCount)+1:rowBorders(rowCount+1), ...
                              colBorders(colmCount)+1:colBorders(colmCount+1));
         histInfo = hist(block(:),nbins);
         tilesHist(:, colmCount, rowCount) = histInfo;
     endfor
 endfor
endfunction