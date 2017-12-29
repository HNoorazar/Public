% This function checks if the subswaps succeeds or not, by checking all
% elements in the associated submatrix.
function s=success(C,D)
eps=2.2204460492503131e-16;
%u=10^(-16);
s=0;
el=length(C);
nc=norm(C);
nd=norm(D);
if (el==3)    
   [Ehat,Fhat,~,~]=subswapg(C,D);
    if (C(2,1)==0)
    nEh=norm(Ehat(3,1:2));
    nFh=norm(Fhat(3,1:2));
    
   elseif (C(2,1)~=0) 
    nEh=norm(Ehat(1,2:3));
    nFh=norm(Fhat(1,2:3));
    end
    
    if ( nEh<eps*nc && nFh<eps*nd)
    %if (Ehat(3,1)<u && Ehat(3,2)<u && Fhat(3,1)<u && Fhat(3,2)<u)
        s=1;
    end
    
    
  elseif (el==4)
    [Ehat,Fhat, ~,~]=subswapg(C,D);
    nEh=norm(Ehat(3:4,1:2));
    nFh=norm(Fhat(3,1:4:2));
    
    if ( nEh<eps*nc && nFh<eps*nd)
    %if (Ehat(3,1)<u && Ehat(3,2)<u && E(4,1)<u && E(4,2)<u && Fhat(3,1)<u && Fhat(3,2)<u && F(4,1)<u && F(4,2)<u)
        s=1;
    end
    
elseif(el==2)
      
      [Ehat,Fhat,~,~]=subswapg(C,D);
      nEh=norm(Ehat(2,1));
      nFh=norm(Fhat(2,1)); 
       if ( nEh<eps*nc && nFh<eps*nd)
        s=1;
       end
end
end