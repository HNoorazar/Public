% This function Makes L such that M=UL or MLinv=U.

function [L,Linv]=ul(M)
n=length(M);
L=eye(n);
Linv=eye(n);
for i=1:n-1
    x=M(n+1-i,1:n+1-i)'; %X's are the rows of matrix M
    alpha=x(n+1-i);
    
    e=zeros(length(x),1);
    e(length(x))=1;
    u=(x-alpha*e)/alpha;
    Ihat=eye(length(x));
    Ghat=Ihat-u*(e');
    
    Gtilda=Ghat;
    gts=length(Gtilda);
    Gtilda(1:gts-1,gts)=-Gtilda(1:gts-1,gts);
    
    Ginv=eye(n);
    Ginv(1:length(x),1:length(x))=transpose(Gtilda); %G is the Matrix such that row vector x'*G=e_n'
    Linv=Ginv*Linv;
    
    G=eye(n);
    G(1:length(x),1:length(x))=transpose(Ghat); %G is the Matrix such that row vector x'*G=e_n'
    
    M=M*G;
    L=L*G; 
   
    
    
end

A=L;
L=Linv;
Linv=A;
end

    
    
