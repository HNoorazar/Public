
% This function takes 3 inputs. (H,U) is Hessenberg-Uppertriangular pencil
% (They are quasi-upper and upper triangular matrices.) and vector v. The
% function transfers v to alpha*e_n defending the structure of (H,U).
% The function Returns (T,S) which in the pencil along with a, which is the
% alpha.

% Length of vector v is the same as H and U sizes.

%RECH=REstart CHase

function [H,U,L,Linv,R,v,G]=rech(H,U,v)
n=length(v);
L=eye(n);
Linv=eye(n);
R=eye(n);
G=eye(n);
for ii=1:n-1
    vp=v(ii:ii+1);
    Gp=ger(vp);
    v(ii)=0;
    H(1:ii+1,ii:ii+1)=H(1:ii+1,ii:ii+1)*Gp;
    U(1:ii+1,ii:ii+1)=U(1:ii+1,ii:ii+1)*Gp;
    
    G(1:ii+1,ii:ii+1)=G(1:ii+1,ii:ii+1)*Gp;
    R(1:ii+1,ii:ii+1)=R(1:ii+1,ii:ii+1)*Gp;
    
    x=U(ii:ii+1,ii);
    [Lp,Linp]=ge1(x);
    U(ii:ii+1,ii:n)=Lp*U(ii:ii+1,ii:n);
    U(ii+1,ii)=0;
    
    if(ii==1)
        H(ii:ii+1,1:n)=Lp*H(ii:ii+1,1:n);
    else
        H(ii:ii+1,ii-1:n)=Lp*H(ii:ii+1,ii-1:n);
    end
    
    L(ii:ii+1,1:ii+1)=Lp*L(ii:ii+1,1:ii+1);
    if(ii+2<n)
        Linv(ii:ii+2,ii:ii+1)=Linv(ii:ii+2,ii:ii+1)*Linp;
    else
        Linv(ii:n,ii:ii+1)=Linv(ii:n,ii:ii+1)*Linp;    
    end
    
    for jj=1:ii-1
        hp=H(ii+2-jj,ii-jj:ii-jj+1);
        Rp=ger(hp);
        
        H(1:ii+1-jj,ii-jj:ii-jj+1)=H(1:ii+1-jj,ii-jj:ii-jj+1)*Rp;
        H(ii+2-jj,ii-jj)=0;
        
        U(1:ii+1-jj,ii-jj:ii-jj+1)=U(1:ii+1-jj,ii-jj:ii-jj+1)*Rp;
        R(1:ii+1,ii-jj:ii-jj+1)=R(1:ii+1,ii-jj:ii-jj+1)*Rp;
        G(1:ii+1,ii-jj:ii-jj+1)=G(1:ii+1,ii-jj:ii-jj+1)*Rp; 
        %v(ii-jj:ii-jj+1)=v(ii-jj:ii-jj+1)*Rp;
        
        x=U(ii-jj:ii-jj+1,ii-jj);
        [Lp,Linp]=ge1(x);
        U(ii-jj:ii-jj+1,ii-jj:n)=Lp*U(ii-jj:ii-jj+1,ii-jj:n);
        U(ii-jj+1,ii-jj)=0;
        
        if (ii-jj==1)
            H(ii-jj:ii-jj+1,1:n)=Lp*H(ii-jj:ii-jj+1,1:n);
        else
            H(ii-jj:ii-jj+1,ii-jj-1:n)=Lp*H(ii-jj:ii-jj+1,ii-jj-1:n);
        end
        
        L(ii-jj:ii-jj+1,1:ii-jj+1)=Lp*L(ii-jj:ii-jj+1,1:ii-jj+1);
        Linv(ii-jj:ii+1,ii-jj:ii-jj+1)=Linv(ii-jj:ii+1,ii-jj:ii-jj+1)*Linp; 
    end
end
end