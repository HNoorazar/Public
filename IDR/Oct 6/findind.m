
% This function takes vector v and natural number k, and returns the index
% of k largest elements of v in ascending (increasing) order (ascending order of indecies).
function vs=findind(v,k)
n=length(v);

[~,I]=sort(abs(v),'descend');
Ik=I(1:k);
vs=sort(Ik,'ascend');
end