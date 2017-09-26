% This function is supposed to find all blocks of different Layers that a
% given pixel belongs to!
% input:
% pixelCoord is a 1-by-2 vector giving coordinate of a given pixel

% tiles is the structure that has the number of layers, borders and centers
% of each Layer. It is output of the function MultiPartitioning.

function  pixelIDs = findIDs(tiles, pixelCoord)
 niteration = size(tiles,2);

%
% initialize the structure containing coordinates of corresponding block.

 pixelIDs(niteration).Layer =  num2str(niteration);
 pixelIDs(niteration).blockCoord = [1,1];

 for count = 1:niteration
     nRowBlocks = size(tiles(count).border,1);
     nColBlocks = size(tiles(count).border,1);
%    
% the following loop determines which row block the pixel belongs to!
%
     for ii=1:nRowBlocks
         if pixelCoord(1) <= tiles(count).border(ii,1)
             pixelIDs(count).Layer =  num2str(count);
             pixelIDs(count).blockCoord(1) = ii;
             break
         endif
     endfor    
%
% the following loop determines which column block the pixel belongs to!
%
     for ii=1:nColBlocks
         if pixelCoord(2) <= tiles(count).border(ii,2)
             pixelIDs(count).Layer =  num2str(count);
             pixelIDs(count).blockCoord(2) = ii;
             break
         endif
     endfor
     
 endfor
endfunction