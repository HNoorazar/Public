% This function returns R,Y,D,Y0 and r which came from
% A*R*Y*D=R*Y0-r*e_n^T. 
% So, r is the last extra column of R which is separately represented.
% k is the number of eigenvalues we are interested in.
% s is taken to be 3.
% Initial guess is always taken to be zero vector.
% Initial residual is random.(Ax=b=r_0)

% This code orthogonalizes each residual with previous 4 residuals.

function [R,Y,D,Y0,P]=IDReigOR1(A,k)
N=length(A);
s=3;
al=floor((2*k)/(s+1))+1;
n=al*(s+1);    % A*R_(n-1)*YD=R_n*Y^0_-

r=rand(N,1);
P=rand(N,s);
P=orth(P);
R=zeros(N,n);
Y=zeros(n-1,n-1);
Y0=zeros(n,n-1);
D=zeros(n-1,n-1);
R(:,1)=r;
nabla_R=zeros(N,s);
% Initiating the residuals, we have already r_0, we need s more residuals
% in G_0.

for ii=1:s
    v=A*r;
    omega=(v'*r)/(v'*v);
    D(ii,ii)=omega;
    nabla_r=-omega*v;
    nabla_R(:,ii)=nabla_r;
    r=r+nabla_r;
    R(:,ii+1)=r;
    Y(ii,ii)=1;
    Y0(ii:ii+1,ii)=[1;-1];
end
nr=s+1;

c=(P'*nabla_R)\(P'*r);
RC=nabla_R*c;
v=r-RC;
t=A*v;
omega=(t'*v)/(t'*t);
for zz=1:4
    D((s+1)+zz-1,(s+1)+zz-1)=omega;
end
nabla_r=-RC-omega*t;
r=r+nabla_r;
y=diff(c);
Y(nr-s:nr,nr)=y;
Y0(nr-s:nr+1,nr)=[y ;-1];
nr=nr+1;
R(:,nr)=r;
   
NRh=zeros(N,s);
NRh(1:N,1:s-1)=nabla_R(1:N,2:s);
NRh(:,s)=nabla_r;
nabla_R=NRh;
    
for kk=1:s
    c=(P'*nabla_R)\(P'*r);
    RC=nabla_R*c;
    v=r-RC;
    t=A*v;
    nabla_r=-RC-omega*t;
    r=r+nabla_r;
    y=diff(c);
    Y(nr-s:nr,nr)=y;
    Y0(nr-s:nr+1,nr)=[y ;-1];
    nr=nr+1;
    R(:,nr)=r;
    
    [r,v]=gram(R(:,nr-kk:nr));
    R(:,nr)=r;
    Y0(nr-kk:nr-1,nr-1)=Y0(nr-kk:nr-1,nr-1)-v;
    
    [r,v]=gram(R(:,nr-kk:nr));
    R(:,nr)=r;
    Y0(nr-kk:nr-1,nr-1)=Y0(nr-kk:nr-1,nr-1)-v;
       
    NRh=zeros(N,s);
    NRh(1:N,1:s-1)=nabla_R(1:N,2:s);
    nabla_r=R(:,nr)-R(:,nr-1);
    NRh(:,s)=nabla_r;
    nabla_R=NRh;
end


for jj=2:al-1
    c=(P'*nabla_R)\(P'*r);
    RC=nabla_R*c;
    v=r-RC;
    t=A*v;
    omega=(t'*v)/(t'*t);
    for zz=1:4
        D(jj*(s+1)+zz-1,jj*(s+1)+zz-1)=omega;
    end
    nabla_r=-RC-omega*t;
    r=r+nabla_r;
    y=diff(c);
    Y(nr-s:nr,nr)=y;
    Y0(nr-s:nr+1,nr)=[y ;-1];
    nr=nr+1;
    R(:,nr)=r;
    
    [r,v]=gramOR1(R(:,nr-4:nr));
    R(:,nr)=r;
    Y0(nr-4:nr-1,nr-1)=Y0(nr-4:nr-1,nr-1)-v;
    
    NRh=zeros(N,s);
    NRh(1:N,1:s-1)=nabla_R(1:N,2:s);
    nabla_r=R(:,nr)-R(:,nr-1);
    NRh(:,s)=nabla_r;
    nabla_R=NRh;
    
    for kk=1:s
        c=(P'*nabla_R)\(P'*r);
        RC=nabla_R*c;
        v=r-RC;
        t=A*v;
        nabla_r=-RC-omega*t;
        rhat=r+nabla_r;
        y=diff(c);
        Y(nr-s:nr,nr)=y;
        
        Rhat=[R(:,nr-3:nr) rhat];
        [r,v]=gramOR1(Rhat);
        
        Y0(nr-s:nr+1,nr)=[y-v ;-1];
        nr=nr+1;
        R(:,nr)=r;
        
        %[r,v]=gramOR1(R(:,nr-4:nr));
        %R(:,nr)=r;
        %Y0(nr-4:nr-1,nr-1)=Y0(nr-4:nr-1,nr-1)-v;
        
        NRh=zeros(N,s);
        NRh(1:N,1:s-1)=nabla_R(1:N,2:s);
        nabla_r=R(:,nr)-R(:,nr-1);
        NRh(:,s)=nabla_r;
        nabla_R=NRh;
    end
end
end


function v=diff(c)
c0=[0;c];
c1=[c;1];
v=c1-c0;
end