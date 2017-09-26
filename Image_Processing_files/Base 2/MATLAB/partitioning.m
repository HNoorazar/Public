% This is a function that used to generate the partion for the original image

% Define the function
function [rowCenters, colCenters, rowBorders, colBorders] = partitioning (matrix, nrowBlocks, ncolBlocks)

% Argument information

% d1: number of the row partition
% d2: number of the column partition 
% rowCenters : row centers of the bin
% colCenters : column centers of the bin
% rowBorders : end points for row partition
% colBorders : end points for column partition

% -----------------------------------------------------
% size of matrix:

nrows = size(matrix,1);
ncol  = size(matrix,2);

q1 = floor(nrows/nrowBlocks);
q2 = floor(ncol/ncolBlocks);
r1 = mod(nrows,nrowBlocks);
r2 = mod(ncol,ncolBlocks);

if (r1==0 && r2==0) 
    rowlength = ones(nrowBlocks,1)*q1;
    columnlength = ones(ncolBlocks,1)*q2;
elseif (r1==0 && r2~=0)
    rowlength = ones(nrowBlocks,1)*q1;
    columnlength = ones(ncolBlocks,1)*q2;
    addspace2 = floor(ncolBlocks/r2);
    idx2 = (addspace2: addspace2: addspace2*r1);
    temp2 = zeros(ncolBlocks,1);
    temp2(idx2) =1;
    columnlength = columnlength+temp2;
elseif (r1~=0 && r2==0)
    columnlength = ones(ncolBlocks,1)*q2;
    rowlength = ones(nrowBlocks,1)*q1;
    addspace1 = floor(nrowBlocks/r1);
    idx1 = (addspace1: addspace1: addspace1*r1);
    temp1 = zeros(nrowBlocks,1);
    temp1(idx1) =1;
    rowlength = rowlength+temp1;
else      
% -----------------------------------------------------
% print out the original side length for each partition bins
     rowlength = ones(nrowBlocks,1)*q1;
     addspace1 = floor(nrowBlocks/r1);
       idx1 = (addspace1: addspace1: addspace1*r1);
       temp1 = zeros(nrowBlocks,1);
       temp1(idx1) =1;
       rowlength = rowlength+temp1;
% ----------------------------------------------------
% Distribute the remainder uniformly to each of the partition
% uniformly distribute row length
% uniformly distribute column length
 columnlength = ones(ncolBlocks,1)*q2;
 addspace2 = floor(ncolBlocks/r2);
 idx2 = (addspace2: addspace2: addspace2*r2);
 temp2 = zeros(ncolBlocks,1);
 temp2(idx2) =1;
 columnlength = columnlength+temp2;
end

% -----------------------------------------------------
% Construct the side length of the partition
rowlength = cumsum(rowlength); 
columnlength = cumsum(columnlength);
% -----------------------------------------------------
% Construct the centers
temprow = [0; rowlength(1:size(rowlength,1)-1,:)];
x = (rowlength + temprow)/2; 

tempcolumn = [0; columnlength(1:size(columnlength,1)-1,:)];
y = (columnlength + tempcolumn)/2; 

% -----------------------------------------------------
% print the results
rowCenters=floor(x);
colCenters=floor(y);
rowBorders = rowlength;
colBorders = columnlength;

end
