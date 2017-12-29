% This function is supposed to take A and B as Hessenberg- Triangular pair
% and transfer them to (H,T) as two quasai-triangular pair using LZE which
% at each step will break A into smaller pieces.

function [A,B,L,Linv,Z]=lzff(A,B)
n=length(A);
L=eye(n);
Linv=eye(n);
Z=eye(n);
s=(n*(n+1))/2;
[c, ~,~, A]=chk(A);

if(c==0)
    [A, B, La, Linva, Za]=lzemod(A,B);
    L=La;
    Z=Za;
    Linv=Linva;
end
[~,l,~]=chqu(A,B);
dummy=0;
while (sum(l)<s)
    
    dummy=dummy+1;
    lm=l(l~=0);
    lml=length(lm);
    
    if(lm(1)==1)
        lm=[lm n+1];
    else
        lm=[0 lm n+1];
    end
    for jj=1:lml
        if (lm(jj+1)-lm(jj)>2)
            a=lm(jj)+1;
            b=lm(jj+1);
            [A(a:b,a:b),B(a:b,a:b),Lp,Linvp,Zp]=lzemod(A(a:b,a:b),B(a:b,a:b));
        
            L(a:b,1:b)=Lp*L(a:b,1:b);
            Z(a:n,a:b)=Z(a:n,a:b)*Zp;
            Linv(a:n,a:b)=Linv(a:n,a:b)*Linvp;
        
            A(a:b,b+1:n)=Lp*A(a:b,b+1:n);
            B(a:b,b+1:n)=Lp*B(a:b,b+1:n);
        
            A(1:a-1,a:b)=A(1:lm(jj),a:b)*Zp;
            B(1:a-1,a:b)=B(1:lm(jj),a:b)*Zp;
            [~,lp,~]=chqu(A(a:b,a:b),B(a:b,a:b));
            
            for kk=1:length(lp)
                if(lp(kk)~=0)
                    lp(kk)=lp(kk)+lm(jj);
                end
            end
            l(a:b)=lp;
       elseif(lm(jj+1)-lm(jj)==2)
           a=lm(jj)+1;
           b=lm(jj+1);
           [A(a:b,a:b),B(a:b,a:b),Lp,Linvp,Zp]=lze2n(A(a:b,a:b),B(a:b,a:b));
        
           L(a:b,1:b)=Lp*L(a:b,1:b);
           Z(a:n,a:b)=Z(a:n,a:b)*Zp;
           Linv(a:n,a:b)=Linv(a:n,a:b)*Linvp;
        
           A(a:b,b+1:n)=Lp*A(a:b,b+1:n);
           B(a:b,b+1:n)=Lp*B(a:b,b+1:n);
        
           A(1:a-1,a:b)=A(1:a-1,a:b)*Zp;
           B(1:a-1,a:b)=B(1:a-1,a:b)*Zp;
           l(lm(jj)+1)=lm(jj)+1;
        end
    end
    [~,l,~]=chqu(A,B);
end
end

