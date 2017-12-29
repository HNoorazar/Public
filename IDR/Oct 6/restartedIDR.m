% OK! I hope this is the last code for getting this job done.
% In this code, I am assuming we are looking for at least 4 eigenvalues to
% have a simple code. (to avoid conflict or hard time in habdling the first s+1 elemenst of matrices!)

% It takes A and number of eigenvalues we are after.

% 1- Run proper steps of IDReig
% 2- Transfer Y and Y0 to upper-quasi triangular pencil.
% 3- cut the last unwanted columns
% 4- step 2 will create a row vector z^t rather than e_n. 
%    So, we transfer the result back to the hessenberg-triangular
%    decomposition of matrix A.
% 5- Run enough steps of IDR again.


% Inputs are A and k. k is the number of wanted eigenvalues.
% Output is v, the vector containing eigenvalues.

function [v]=restartedIDR(A,k)

N=length(A);
[R,Y,D,Y0,P]=IDReig(A,k);
s=3;
al=floor((2*k)/(s+1))+1;
n=al*(s+1);
sY=length(Y);
YD=Y*D;
[~,qq]=size(R);
r_last=R(:,qq);
R=R(1:N,1:sY);
Y0=Y0(1:sY,1:sY);
[H,U,~,Linv,Z]=lzff(Y0,YD);
zt=Z(sY,:);
R=R*Linv;
kh=floor(k/(s+1))+1;
kh=kh*(s+1)-1;
[Y0,YD,IYtinv,~,IXt]=move(Y0,YD,kh);
R=R*IYtinv;
zt=zt*IXt;


tol=floor((n-kh)/(s+1));
ss=0;
zz=0;
while (ss==0 && zz<tol)
    if (Y0(kh+zz*(s+1)+1,kh+zz*(s+1))==0)
        Y0=Y0(1:kh+zz*(s+1),1:kh+zz*(s+1));
        YD=YD(1:kh+zz*(s+1),1:kh+zz*(s+1));
        R=R(1:N,1:kh+zz*(s+1));
        zt=zt(1:kh+zz*(s+1));
        ss=1;
    else
        zz=zz+1;
    end
end

[Y0h,YDh,~,Linv,~,zt,~]=rech(Y0,YD,zt);
R=R*Linv;
szt=length(zt);
r_last=r_last*zt(szt);
% zt(szt)=1;
R=[R r_last];
[~,ncR]=size(R); % ncR is number of columns of R.
Y=zeros(n-1,n-1);
Y0=zeros(n,n-1);
D=zeros(n-1,n-1);

D(1:szt,1:szt)=eye(szt);
Y(1:szt,1:szt)=YDh;
Y0(1:szt,1:szt)=Y0h;
Y0(szt+1,szt)=-1;

r=r_last;
nabla_R=zeros(N,s);
nr=ncR;
for zz=1:s
    nabla_R(:,s-zz+1)=R(:,ncR-zz+1)-R(:,ncR-zz);
end
    
for jj=ncr+1:n
    
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
    NRh=zeros(N,s);
    NRh(1:N,1:s-1)=nabla_R(1:N,2:s);
    NRh(:,s)=nabla_r;
    nabla_R=NRh;
    for zz=1:s
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
        NRh=zeros(N,s);
        NRh(1:N,1:s-1)=nabla_R(1:N,2:s);
        NRh(:,s)=nabla_r;
        nabla_R=NRh;
    end
end
v=eig(Y0,Y*D);
v=sort(v);
end