    

function signature = findSignatures(allHists, tiles, pixelIDs)

% allHists contans all histogram informations for all partitions of all
% sizes. It is output of the function wholeHistInfo.
% allHists.Name and allHists.histos
%
%
% tiles contains all borders and centers of all the partitions of all
% sizes. It is output of the function MultiPartitioning.
%
% tiles.Layer   
% tiles.border
% tiles.centers

nbins = size(allHists(1).histos,1);
nLayers = size(tiles,2);
signature = zeros(nbins, nLayers);

if size(allHists) ~= size(tiles)
    error('size of allHists and tiles does not go together!')
end


for ii=1:nLayers
    coord = pixelIDs(ii).blockCoord;
    signature(:,ii) = allHists(ii).histos(:, coord(2), coord(1)) ;
end

end
