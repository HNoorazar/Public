
N=10;
s=3; 
P=rand(N,s);
P=orth(P);
A=rand(N,N);
b=rand(N,1);
r=b;     %initial residual
R=zeros(N,s+1);
delta_R=zeros(N,s);
R(:,1)=r;
D=zeros(16,1);
%intial residuals in g_0

for n=1:s
    
    v=A*r;
    omega = (v'*r)/(v'*v);
    if omega<0.00001
        omega=omega+1;
    end
    D(n,n)=omega;
    delta_r = -omega*v;
    delta_R(:,n)= delta_r;
    r = r + delta_r;
    R(:,n+1)=r;
    
end

%starting part of Hessenberg
H=eye(s);
H(s+1,s)=-1;
for i=1:s-1
    H(i+1,i)=-1;
end

%end of Starting of Hessenberg
n=s+2;
j=1;

for k=1:16
    c=(transpose(P)*delta_R)\(transpose(P)*r);
    v=r-delta_R*c;
    t=A*v;
    omega=transpose(t)*v/norm(t);
    if omega<0.00001
        omega=omega+1;
    end
    D(k)=omega;
    delta_r=-delta_R*c-omega*A*v;
    r=r+delta_r;
    n=n+1;
    for t=1:s-1 
        delta_R(:,t)= delta_R(:,t+1);
    end
    delta_R(:,s)=delta_r;
  
    for l=0:s
        c=(transpose(P)*delta_R)\(transpose(P)*r);
        v=r-delta_R*c;
        delta_r=-delta_R*c-omega*A*v;
        r=r+delta_r;
        n=n+1;
        
        for t=1:s-1 
        delta_R(:,t)= delta_R(:,t+1);
        end
        delta_R(:,s)=delta_r;
        j=j+1;
    end
   
    
end
