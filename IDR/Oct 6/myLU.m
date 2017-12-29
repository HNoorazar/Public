% This function will give LU decomposition of M where M is non-square such
% that M=LU. M is m-by-n and m>n here.

function [L, Linv, U]=myLU(M)
[m,n]=size(M);
L=eye(m);
Linv=eye(m);

for ii=1:n
    
    v=M(ii:m,ii);
    [G,Ginv]=ge1(v);
    
    M(ii+1:m,ii)= M(ii,ii)*G(2:m-ii+1,1)+M(ii+1:m,ii);
    M(ii+1:m,ii+1:n)=G(2:m-ii+1,1)*M(ii,ii+1:n)+M(ii+1:m,ii+1:n);
    
    if (ii==1)
    
        L(2:m,1)=G(2:m,1);
        Linv(2:m,1)=Ginv(2:m,1);
        
    else
        
   %updating L_(2,1)
      L(ii+1:m,1)=G(2:m-ii+1,1)*L(ii,1)+L(ii+1:m,1);
      L(ii+1:m,2:ii-1)=G(2:m-ii+1,1)*L(ii,2:ii-1)+L(ii+1:m,2:ii-1);
      
   %updating L(2,2)
      L(ii+1:m,ii)=G(2:m-ii+1);
    
    %Updating Linv
    Linv(ii+1:m,ii)=Ginv(2:m-ii+1,1);
    
        
    end
 
end
U=M;
H=Linv;
Linv=L;
L=H;

end
