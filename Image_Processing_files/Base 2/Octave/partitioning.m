% This is a function that used to generate the partion for the original image

function [rowCenters, colCenters, rowBorders, colBorders] = partitioning (nrow, ncol, ...
                                                                           nrowPartitions, ncolPartitions)
  % Argument information
  % nrow: Number of Rows of A
  % ncol Number of Columns of A 
  % nrowPartitions
  % ncolPartitions
  
  % -----------------------------------------------------
  % Find the quotient and remainder of p/d1 and 1/d2
  q1 = floor(nrow/nrowPartitions);
  q2 = floor(ncol/ncolPartitions);
  r1 = mod(nrow,nrowPartitions);
  r2 = mod(ncol,ncolPartitions);
  
  if (r1==0 && r2==0) 
         rowlength = ones(nrowPartitions,1)*q1;
         columnlength = ones(ncolPartitions,1)*q2;
  elseif (r1==0 && r2~=0)
         rowlength = ones(nrowPartitions,1)*q1;
         columnlength = ones(ncolPartitions,1)*q2;
         addspace2 = floor(ncolPartitions/r2);
         idx2 = (addspace2: addspace2: addspace2*r1);
         temp2 = zeros(ncolPartitions,1);
         temp2(idx2) =1;
         columnlength = columnlength+temp2;
  elseif (r1 ~= 0 && r2==0)
          columnlength = ones(ncolPartitions,1)*q2;
          rowlength = ones(nrowPartitions,1)*q1;
          addspace1 = floor(nrowPartitions/r1);
          idx1 = (addspace1: addspace1: addspace1*r1);
          temp1 = zeros(nrowPartitions,1);
          temp1(idx1) =1;
          rowlength = rowlength+temp1;
  else      
% -----------------------------------------------------
% print out the original side length for each partition bins
       rowlength = ones(nrowPartitions,1)*q1;
       columnlength = ones(ncolPartitions,1)*q2;
       addspace1 = floor(nrowPartitions/r1);
       idx1 = (addspace1: addspace1: addspace1*r1);
       temp1 = zeros(nrowPartitions,1);
       temp1(idx1) =1;
       rowlength = rowlength+temp1;

  % ----------------------------------------------------
  % Distribute the remainder uniformly to each of the partition
  % uniformly distribute row length
 
  % uniformly distribute column length
  addspace2 = floor(ncolPartitions/r2);
  idx2 = (addspace2: addspace2: addspace2*r2);
  temp2 = zeros(ncolPartitions,1);
  temp2(idx2) =1;
  columnlength = columnlength + temp2;
  endif

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
    rowCenters = x;
    colCenters = y; 
    rowBorders = rowlength;
    colBorders = columnlength;
  endfunction