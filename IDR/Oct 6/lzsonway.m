% The difference between this and LZSmod is that this one has to return L
% as well
% LZS does one step of LZ algorithm (LZ is supposed to transfer (H,U)
% Hess-Upp pencil to (T,S) two quasi-upper.

%%%% $$$$ Zeros are set manually.$$$$$ %%%%
function [T, S, L, Linv, Z]=lzsonway(A,B)
n=length(A);
m=2;
s=3;
% Zcheck=eye(n);   
L=eye(n);
Linv=eye(n);
Z=eye(n);
T=A;
S=B;
[sh1,sh2]=myeig(T(n-1:n,n-1:n),S(n-1:n,n-1:n));
x=sv(T,S,sh1,sh2);
%x=sv1s(T,S);
xhat=real(x(1:m+1));    % Elimination matrix for first column of p(AB^-1)
[Gtilda,Gtinv]=ge1(xhat);

T(1:m+1,1:n)=Gtilda*T(1:m+1,1:n); 
S(1:m+1,1:n)=Gtilda*S(1:m+1,1:n);
L(1:m+1,1:m+1)=Gtilda;
Linv(1:m+1,1:m+1)=Gtinv;
v=eig(A,B);
v=sort(v);
for ii=1:n-m-1   
    [~,W]=ul(S(ii:m+ii,ii:m+ii));
    W(3,2)=0;
    S(1:ii,ii) = S(1:ii,ii:m+ii)*W(:,1);
    S(ii+1,ii)=0; S(ii+2,ii)=0;
    
    T(1:m+ii+1,ii)= T(1:m+ii+1,ii:m+ii)*W(:,1);
    %ii
    %T
    Z(ii:n,ii)=Z(ii:n,ii:m+ii)*W(:,1); 
    w=eig(T,S);
    w=sort(w);
    normm=norm(v-w);
    
    %------ Elimination on A(Or T which is Hessenberg) 
    x=T(ii+1:m+ii+1,ii);
    [Gtilda,Gtinv]=ge1(x);
    
    T(ii+2:m+ii+1,ii)=zeros(m,1);
    T(ii+1:m+ii+1,ii+1:n)=  Gtilda*T(ii+1:m+ii+1,ii+1:n);
    %ii
    %T
    S(ii+1:m+ii+1,ii+1:n)=  Gtilda*S(ii+1:m+ii+1,ii+1:n);
    w=eig(T,S);
    w=sort(w);
    normm=norm(v-w);
    L(ii+1:m+ii+1,ii+1)= Gtilda(:,1);
    L(ii+1:m+ii+1,1:ii)= Gtilda*L(ii+1:m+ii+1,1:ii);
    Linv(ii+1:n,ii+1)  = Linv(ii+1:n,ii+1:ii+m+1)*Gtinv(:,1);    
end

for ii=n-m:n-1
    [~,W]=ul(S(ii:n,ii:n));
    
    S(1:ii,ii)=S(1:ii,ii:n)*W(:,1);
    S(ii+1:n,ii)=zeros(n-ii,1);

    T(1:n,ii)=T(1:n,ii:n)*W(:,1);
    Z(ii:n,ii)=Z(ii:n,ii:n)*W(:,1);
    %ii
    %T
    if (ii<n-1)

        x=T(ii+1:n,ii);
        [Gtilda,Gtinv]=ge1(x);
        
        T(ii+1:n,ii+1:n)=Gtilda*T(ii+1:n,ii+1:n);
        T(n,n-2)=0;
        S(ii+1:n,ii:n)=Gtilda*S(ii+1:n,ii:n);
        %ii
        %T
        L(n,n-1)=Gtilda(2,1);
        L(n-1:n,1:n-2)=Gtilda*L(n-1:n,1:n-2);
        Linv(1:n,ii+1:n)=Linv(1:n,ii+1:n)*Gtinv;  
    end
end
end