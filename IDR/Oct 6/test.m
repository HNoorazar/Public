function [y]=test(x)
k=length(x);
alpha=norm(x,inf);
p=fp(x);

x(p)=x(1);

x(1)=alpha;
y=x;
p
end

function [p]=fp(x)
alpha=norm(x,inf);
for ii=1:length(x)
if (x(ii)==alpha)
    p=ii;

end
end
end
