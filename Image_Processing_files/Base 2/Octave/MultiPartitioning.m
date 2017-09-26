% This function hold all border and centroid informations for all of the
% partition sizes.

function allTiles = MultiPartitioning(imageMatrix, base)
 
 % input:
 % imageMatrix is matrix of image read by MATLAB
 % Base is the way we want to partition the image, diatically or etc.
 
 % outout:
 % tile is the structure which has all the borders and cener node
 % information in it for partitions of different sizes.
 %-------------- Number of interations here:
 [nRow, nCol] = size(imageMatrix);
 if base == 2
     noIterations = log2(min(nRow,nCol));
 else
     noIterations = log(min(nRow,nCol))/log(base);
 endif

 noIterations = floor(noIterations);

%
% Initialize the structure here, the last one is just for making the str.
%
 allTiles(noIterations).Layer =  num2str(noIterations);
 allTiles(noIterations).border = 1;
 allTiles(noIterations).centers = 1;

 for count = 1:noIterations
     nRowPartition = base^count;
     nColPartition = base^count; 
     [rowCenters, colCenters, rowBorders, colBorders] = partitioning (nRow, nCol, nRowPartition, nColPartition);
     centerCoord = [rowCenters, colCenters];
     borders = [rowBorders, colBorders];
     allTiles(count).Layer = strcat('Layer', num2str(count));
     allTiles(count).border = borders;
     allTiles(count).centers = centerCoord;
    % 100 tile1.Name = strcat('',num2str(ii));
    % 100 tile1.info = {centerCoord, borders}
 endfor
endfunction