
% This function will do one step of restart. after IDR is done, it takes
% the residual matrix and Y and Y0, then transfers (Y0,Y) to (T,S) by LZ
% algorithm, then does one step of restart which is transfering z^T to e_n^T
% and also changes in R, Y, Y0.(hrs=Hossein's Restart,one,Step)

function [Rnew, Ynew, Yhnew]=hrs(R, Y, Yh)
n=length(Y);

[T, S, Linv, Z]=lzf(Yh,Y);
Rnew=R*Linv;
zt= Z(n,1:n);

for jj=1:n-1
    G=er(zt(jj,jj+1));  %G is elimination Mateix acting on zt to transfer it to e_n^t
    S(1:jj+2,jj:jj+1)=S(1:jj+2,jj:jj+1)*G;
    T(1:jj+2,jj:jj+1)=T(1:jj+2,jj:jj+1)*G;
    
    %now we have to kill the effect of G on T and S.
for kk=1:jj
     
    [GG, GGinv]=el(S(jj-kk+1:jj-k+2,jj-kk+1));
    S(jj-kk+1:jj-k+2,jj-kk:n)=GG*S(jj-kk+1:jj-k+2,jj-kk:n); % L is applied to S and T
    T(jj-kk+1:jj-k+2,jj-kk:n)=GG*T(jj-kk+1:jj-k+2,jj-kk:n);
    
    Rnew(1:n,jj-kk+1:jj-k+2)=Rnew(1:n,jj-kk+1:jj-k+2)*GGinv;      % Linv is applied to R(=residual matrix)
    
    if(jj-kk>0)
        
        RR=er(T(jj-kk+2,jj-kk:jj-kk+1));
        T(1:jj-kk+2,jj-kk:jj-kk+1)=T(1:jj-kk+2,jj-kk:jj-kk+1)*RR;
        S(1:jj-kk+2,jj-kk:jj-kk+1)=S(1:jj-kk+2,jj-kk:jj-kk+1)*RR;
        
    end
    
end
end

Ynew=S;
Yhnew=T;



 
end

%----------------------------------------------------------------
% Elimination Matrix for a column vector of size 2(Applied on left)
% This eliminates the second component.
% Elimination matrix will be Gauss transform.
% el = Elimination on Left

function [GG, GGinv]=el(v)
GG=eye(2,2);
GGinv=eye(2,2);
if (v(2)==0) 
    return
else
 %GG=eye(2,2);
 GG(2,1)=-(v(2)/v(1));
 
 %GGinv=eye(2,2);
 GGinv(2,1)=v(2)/v(1);
end

end
%----------------------------------------------------------------
% Elimination Matrix for a row vector of size 2 (Applied on right)
% This eliminates the second component.
% Elimination matrix will be Gauss transform.
% er = Elimination on Right
function [RR]=er(v)
RRt=eye(2,2);

if (v(2)==0)
    return
 else
    RRt(1,2)=-(v(1)/v(2));
 end
RR=transpose(RRt);

end
