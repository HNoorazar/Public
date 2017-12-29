
% This function will make 4 residuals with GMRES Method

function [V,H]=GMRESidf(A)
n=length(A);
r=rand(n,1);
R(:,1)=r;
beta=norm(r);
v=r/beta;
H=[];
V(:,1)=v;
[~,m]=size(V);

for kk=1:3
    
    for jj=1:m
        w=A*V(:,jj);
        for ii=1:jj
            H(ii,jj)=dot(w,V(:,ii));
            w=w-H(ii,jj)*V(:,ii);
        end
        H(jj+1,jj)=norm(w);
        V(:,jj+1)=w/H(jj+1,jj);
    end
    %e=zeros(m+1,1);
    %e(1)=beta;
    %y=H\e;
    %x=V*y;
    %r=R(:,1)-A*V*y;
    %R(:,jj)=r;
    [~,m]=size(V);
end
end