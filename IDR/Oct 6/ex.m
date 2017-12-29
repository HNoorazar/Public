% This function extracts nonzero elements of a vector

function nz=ex(v)
m=length(v);
nz=zeros(1,1);
c=0;
for ii=1:m
    if v(ii)~=0
       c=c+1;
       nz(c)=v(ii);
    end
end
end


