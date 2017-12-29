format long

N=10;
s=3;
maxiter=40;
k=100;
A=rand(N,N);
P=rand(N,s);   %
b=rand(N,1);
x=zeros(N,1);  %initial guess
r=b;           %initial residual 
tol=0.000000001;
X(:,1)=x;
R(:,1)=r;

for j=1:k
    
    w=A*V(:,j);
  
    for i=1:j
        h(i,j)=V(:,i)'*w;
        w=w-h(i,j)*V(:,i);
        H(i,j)=h(i,j);
    end
    h(j+1,j)=norm(w);
    if h(j+1,j)==0
        fprintf (danger);
    else 
        H(j+1,j)=h(j+1,j);
        V(:,j+1)=w/h(j+1,j);
    end
   H;
    y=H\(beta*eye(j+1,1));
    x=V(1:n,1:j)*y;
    r=b-A*x;
    R(:,j)=r;
end
