% This function swaps blocks of (E,F) where (E,F) is a block triangular
% pencil.

function [Ehat,Fhat, IY,IX]=subswapg(E,F)
Es=length(E);
% IR2=zeros(2,2); IR2(1,2)=1; IR2(2,1)=1;
% IR4=zeros(4,4); IR4(1:2,3:4)=eye(2); IR4(3:4,1:2)=eye(2);
% IRL31=zeros(3,3); IRL31(1:2,2:3)=eye(2); IRL31(3,1)=1; IRR31=transpose(IRL31);
% IRL32=zeros(3,3); IRL32(1,3)=1; IRL32(2:3,1:2)=eye(2); IRR32=transpose(IRL32);

%Forming blocks here:
if (Es==4)
   
   E11=E(1:2,1:2);
   E22=E(3:4,3:4);
   E12=E(1:2,3:4);
  
   F11=F(1:2,1:2);
   F22=F(3:4,3:4);
   F12=F(1:2,3:4);
end

if(Es==3)
    
    if (E(3,1)==0 && E(3,2)==0)
        E11=E(1:2,1:2);
        E22=E(3,3);
        E12=E(1:2,3);
        
        F11=F(1:2,1:2);
        F22=F(3,3);
        F12=F(1:2,3);
    
    elseif (E(2,1)==0 && E(3,1)==0)
        
        E11=E(1,1);
        E22=E(2:3,2:3);
        E12=E(1,2:3);
        
        
        F11=F(1,1);
        F22=F(2:3,2:3);
        F12=F(1,2:3);
        
    end
    
end

if(Es==2)
  E11=E(1,1);
  E12=E(1,2);
  E22=E(2,2);
  
  F11=F(1,1);
  F12=F(1,2);
  F22=F(2,2);
end

%--- End of forming Blocks
%E
m=length(E11);
k=length(E22);

Im=eye(m);
Ik=eye(k);

A=[kron(Ik,E11) kron(transpose(E22),Im); kron(Ik,F11) kron(transpose(F22),Im)];
b=-[vec(E12);vec(F12)];
XY=A\b;
X=zeros(m,k); 
Y=zeros(m,k);

for ii=1:k
X(:,ii)=XY(m*(ii-1)+1:m*ii);
end

for ii=1:k
Y(:,ii)=XY(m*(k+ii-1)+1:(k+ii)*m);
end


IX=eye(m+k,m+k);
IX(k+1:m+k,1:k)=X;

IY=eye(m+k,m+k);
IY(k+1:m+k,1:k)=Y;

%[Q,R]=myLUs(IX);
%[Z,Linv]=ul(IY);

% EN=[E22 zeros(k,m);E12 E11];
% FN=[F22 zeros(k,m);F12 F11];
% Ehat=IY*EN*IX;
% Fhat=IY*FN*IX;
[a,b]=size(E22);
[c,d]=size(E11);

Ehat=[E22 zeros(k,m);zeros(c,b) E11];
Fhat=[F22 zeros(k,m);zeros(c,b) F11];

end