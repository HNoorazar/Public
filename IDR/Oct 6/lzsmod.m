% The difference between this and LZSmod is that this one has to return L
% as well
% LZS does one step of LZ algorithm (LZ is supposed to transfer (H,U)
% Hess-Upp pencil to (T,S) two quasi-upper.
function [T, S, L, Linv, Z]=lzsmod(A,B)
n=length(A);
m=2;  
% Zcheck=eye(n);   
L=eye(n);
Linv=eye(n);
Z=eye(n);
T=A;
S=B;
[e1,e2]=myeig(T(n-1:n,n-1:n),S(n-1:n,n-1:n));
x=sv(T,S,e1,e2);
xhat=real(x(1:m+1));
[Gtilda,Gtinv]=ge1(xhat);

L(1:m+1,1:m+1)=Gtilda;
Linv(1:m+1,1:m+1)=Gtinv;

T(1:m+1,1:n)=Gtilda*T(1:m+1,1:n);
S(1:m+1,1:n)=Gtilda*S(1:m+1,1:n);
for ii=1:n-m-1

 [~,W]=ul(S(ii:m+ii,ii:m+ii));
 H=eye(m+1);
 H(:,1)=W(:,1);
 
 Z(ii:n,ii)=Z(ii:n,ii:m+ii)*H(:,1);
 S(1:m+ii,ii) = S(1:m+ii,ii:m+ii)*H(:,1);
 T(1:m+ii+1,ii)= T(1:m+ii+1,ii:m+ii)*H(:,1);
 %------ Elimination on A(Or T which is Hessenberg) 
 x=T(ii+1:m+ii+1,ii);
 [Gtilda,Gtinv]=ge1(x);

 L(ii+1:ii+m+1,1:ii+m+1)=Gtilda*L(ii+1:ii+m+1,1:ii+m+1);
 Linv(ii+1:n,ii+1)=Linv(ii+1:n,ii+1:ii+m+1)*Gtinv(:,1);
 T(ii+1:m+ii+1,ii:n)=       Gtilda*T(ii+1:m+ii+1,ii:n); % At first this part was written as T(ii+1:m+ii+1,1:n)
 S(ii+1:m+ii+1,ii:n)=       Gtilda*S(ii+1:m+ii+1,ii:n);
end
for ii=n-m:n-1
 
    [~,W]=ul(S(ii:n,ii:n));
    H=eye(n-ii+1);
    H(:,1)=W(:,1);
        
    S(1:n,ii)=S(1:n,ii:n)*W(:,1);
    
    T(1:n,ii)=T(1:n,ii:n)*W(:,1);
    Z(ii:n,ii)=Z(ii:n,ii:n)*H(:,1);
    if (ii<n-1)
        x=T(ii+1:n,ii);
        [Gtilda,Gtinv]=ge1(x);
        L(ii+1:n,1:n)=Gtilda*L(ii+1:n,1:n);
        Linv(1:n,ii+1:n)=Linv(1:n,ii+1:n)*Gtinv;
        T(ii+1:n,ii:n)=Gtilda*T(ii+1:n,ii:n);
        S(ii+1:n,ii:n)=Gtilda*S(ii+1:n,ii:n);
    end 
end
end