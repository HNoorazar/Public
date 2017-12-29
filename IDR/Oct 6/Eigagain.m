clear all;
clc;
format long
N=20; %size of Matrix A
s=3; 
k=10; % We are looking for 10 largest eigenvalue
j=k;  % to restart we need to take j=k extra steps
n=k+j;% number of steps
R=zeros(N,n+1);
delta_R=zeros(N,s);
delta_r=zeros(N,1);

Y=zeros(n,n);
H=zeros(n,n);

D=zeros(n,1);
D(1:s)=ones(s,1);

P=rand(N,s);         %Basis for S_purp
P=orth(P);

A=rand(N,N);
b=rand(N,1);
r=b;            %initian guess iz zero, so initial residual is b
R(:,1)=r;
%starting part
for ii=1:s
    v=A*r;
    omega=dot(v,r)/dot(v,v);
    delta_r=-omega*v;
    r=r+delta_r;
    R(:,ii+1)=r;
    delta_R(:,ii)=delta_r;
    Y(ii,ii)=omega;
end
% We are good as of now (Checked)
nr=s+1;



for ii=1:floor((n+1)/(s+1))-1
    c=(P'*delta_R)\(P'*r);
    y=[c;1]-[0;c];
    delta_Rc= delta_R*c;
    v=r-delta_Rc;
    t=A*v;
    omega=dot(t,v)/dot(t,t);
    D(ii*(s+1):(ii+1)*(s+1)-1)=omega;
    Y(nr-s:nr,nr)=y;
    delta_r=-delta_Rc-omega*t;
    r=r+delta_r;
    nr=nr+1;
    R(:,nr)=r;
    
    for k=1:s-1           %updating delta_R 
        delta_R(:,k)=delta_R(:,k+1);
    end
    delta_R(:,s)=delta_r;
    
    for z=1:s
        c=(P'*delta_R)\(P'*r);
        y=[c;1]-[0;c];
        Y(nr-s:nr,nr)=y;
        delta_Rc= delta_R*c;
        v=r-delta_Rc;
        t=A*v;
        delta_r=-delta_Rc-omega*t;
        r=r+delta_r;
        nr=nr+1;
        R(:,nr)=r; 
        for k=1:s-1           %updating delta_R 
        delta_R;
        delta_R(:,k)=delta_R(:,k+1);
        end
        delta_R(:,s)=delta_r;
    end    
end


%last residual (21st residual)
c=(P'*delta_R)\(P'*r);
y=[c;1]-[0;c];
Y(17:20,20)=y;
delta_Rc= delta_R*c;
v=r-delta_Rc;
t=A*v;
omega=dot(t,v)/dot(t,t);
D(n)=omega;
Y(nr-s:nr,nr)=y;
delta_r=-delta_Rc-omega*t;
r=r+delta_r;
nr=nr+1;
R(:,nr)=r;

D=diag(D);
Rn=R(1:end,1:n);

%defining Hessenberg-matrix in Pencil
H=Y-diag(ones(n-1,1),-1);
for ii=1:s
    H(ii,ii)=1;
    H(ii+1,ii)=-1;
end
I=eye(n,n);
en=I(:,n);
e1=I(:,1);
YD=Y*D;