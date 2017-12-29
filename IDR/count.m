%This function counts the number of nonzero components of a vector

function c=count(v)
l=length(v);
c=0;
for ii=1:l
    if (v(ii)~=0)
        c=c+1;
    end
    
end
