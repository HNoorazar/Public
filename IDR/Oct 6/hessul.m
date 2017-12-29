% Transfering (A,B) to (H,T) where (H,T) is Hess-Triang by second Algorithm

function [A,B,G,Ginv,H]=hessul(A,B)
n=length(A);
G=eye(n);
Ginv=eye(n);
H=eye(n);
for ii=1:n-1
    
    [~,W]=ul(B(ii:n,ii:n));
    W(2:length(W),2:length(W))=eye(n-ii);
    H(ii:n,ii)=W(:,1);
    B(1:ii,ii)=B(1:ii,ii:n)*W(:,1);
    B(ii+1:n,ii)=zeros(n-ii,1);
    A(:,ii)=A(:,ii:n)*W(:,1);
    
    v=A(ii+1:n,ii);
    [Gp,Ginp]=ge1(v);
    A(ii+1:n,ii+1:n)=Gp*A(ii+1:n,ii+1:n);
    A(ii+2:n,ii)=zeros(n-ii-1,1);
    B(ii+1:n,ii+1:n)=Gp*B(ii+1:n,ii+1:n);
    G(ii+2:n,ii+1)=Gp(2:length(Gp),1);
    G(ii+1:n,2:ii)=Gp*G(ii+1:n,2:ii);
    
end

end