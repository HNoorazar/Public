function [c, lh, lc, A]=chk(A)
% checks sub diagonal entries to set them zero if small enough
% Returns locations of zeros in order- if we have more than one small
% subdiagonal. 
% lc(=location counting) gives the number of qualified elements to be set to zero
% lh gives locations of zeros

n=length(A);
u=10^(-16);

lc=0;         %location counts or zero count
lh=zeros(1,1);
c=0;


for jj=1:n-1
    a1=abs(A(jj,jj));
    a2=abs(A(jj+1,jj+1));
    a3=abs(A(jj+1,jj));
    
    if (a3<u*(a1+a2))
      A(jj+1,jj)=0;
    end
end

for jj=1:n-1
    if(A(jj+1,jj)==0)
    lc=lc+1;
    lh(lc)=jj;
    end
end

if (lc>0)
    c=1;
end

end

