%Obviosly this function is LU decomposition, witout any row changes where
%Matrix M is square. L and Linv are labled such that A=LU

function [L,Linv,U]=myLUs(M)
n=length(M);
L=eye(n);
Linv=eye(n);
U=eye(n);

for ii=1:n-1
v=M(ii:n,ii);

[G,Ginv]=ge1(v);

M(ii+1:n,ii)=M(ii,ii)*G(2:n-ii+1,1)+M(ii+1:n,ii); %zeros(n-ii,1)
M(ii+1:n,ii+1:n)=G(2:n-ii+1,1)*M(ii,ii+1:n)+M(ii+1:n,ii+1:n);


if(ii==1)
L(2:n,1)=G(2:n,1);
Linv(2:n,1)=Ginv(2:n,1);
else
  
    %updating L_(2,1)
    L(ii+1:n,1)=G(2:n-ii+1,1)*L(ii,1)+L(ii+1:n,1);
    L(ii+1:n,2:ii-1)=G(2:n-ii+1,1)*L(ii,2:ii-1)+L(ii+1:n,2:ii-1);
    
    %updating L(2,2)
    L(ii+1:n,ii)=G(2:n-ii+1);
    
    %Updating Linv
    Linv(ii+1:n,ii)=Ginv(2:n-ii+1,1);
    
end
    
end
U=M;
H=Linv;
Linv=L;
L=H;


end