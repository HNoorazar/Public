% This function takes (T,S) where T is quasi triangular and S is upper
% triangular, and then moves the biggest eigenvalues to top.
% Here we are looking for k largest eigenvalues.

function [T,S,IYtinv,IYt,IXt]=move(T,S,k)
n=length(T);
e=eig(T,S);
%v=zeros(1,n);
vs=findind(e,k);
IR2=zeros(2,2); IR2(1,2)=1; IR2(2,1)=1;
IR4=zeros(4,4); IR4(1:2,3:4)=eye(2); IR4(3:4,1:2)=eye(2);
IRL31=zeros(3,3); IRL31(1:2,2:3)=eye(2); IRL31(3,1)=1; IRR31=transpose(IRL31);
IRL32=zeros(3,3); IRL32(1,3)=1; IRL32(2:3,1:2)=eye(2); IRR32=transpose(IRL32);

tmbs=0;   % Total size of moved blocks gatherd up top
ii=1;
%pp=0;
IYt=eye(n);
IYtinv=eye(n);
IXt=eye(n);
while(tmbs<k && ii<k+1)
    zc=0;                                    % zc stands for zero counting,  
                                             % It counts the number of
                                             % rejected swaps.
                                             % zcl will be a vector
                                             % containing locations of
                                             % rejected swaps.
    zcl=zeros(1,1);
    
    if (imag(e(vs(ii)))==0)
        pp=ii+1;
    else
        pp=ii+2;
    end
    cv=comov(T(tmbs+1:vs(ii),tmbs+1:vs(ii))); % Counts the number of non-zero 
                                              % elemets of subdiagonal.
    bkc=0;                                    % This is counting backwards.
                                              % We are moving a block from
                                              % a position to the top. So,
                                              % we need to count how many
                                              % steps we are taing back to
                                              % see where we are at each
                                              % moment.
   
   if (vs(ii)<n)
      for jj=tmbs+1:vs(ii)-cv-1
       if(T(vs(ii)-bkc-1,vs(ii)-bkc-2)~=0 && T(vs(ii)-bkc+1,vs(ii)-bkc)~=0) 
             E=T(vs(ii)-bkc:vs(ii)-bkc+3,vs(ii)-bkc:vs(ii)-bkc+3);
             F=S(vs(ii)-bkc:vs(ii)-bkc+3,vs(ii)-bkc:vs(ii)-bkc+3);
             s=success(E,F);
%             sprintf('m1')
            % T
             if (s==1)
                 [~,~, IY,IX]=subswapg(E,F);
                 IY=IY*IR4;
                 IX=IR4*IX;
                 IYinv=[-IY(3:4,3:4) eye(2); eye(2) zeros(2,2)];
                 
                 aa=vs(ii)-bkc;
                 bb=vs(ii)-bkc+3;
                 
                 IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
                 IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
                 IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
                 
                 T(aa:bb,aa:n) = IY*T(aa:bb,aa:n);
                 S(aa:bb,vs(ii)-bkc:n) = IY*S(aa:bb,aa:n);
             
                 T(1:bb,aa:bb) = T(1:bb,aa:bb)*IX;
                 S(1:bb,vs(ii)-bkc:vs(ii)-bkc+3) = S(1:bb,aa:bb)*IX;
                 T(aa:aa+1,aa-2:aa-1)=zeros(2,2);
                 S(aa:aa+1,aa-2:aa-1)=zeros(2,2);
                 
             elseif (s==0)
                 zc=zc+1;
                 zcl(zc)=vs(ii)-bkc;
             end
             bkc=bkc+2;
        elseif (T(vs(ii)-bkc-1,vs(ii)-bkc-2)== 0 && T(vs(ii)-bkc+1,vs(ii)-bkc)~= 0)
            E=T(vs(ii)-bkc:vs(ii)-bkc+2,vs(ii)-bkc:vs(ii)-bkc+2);
            F=S(vs(ii)-bkc:vs(ii)-bkc+2,vs(ii)-bkc:vs(ii)-bkc+2);
            %sprintf('m2')
            %T
            s=success(E,F);
            
            if (s==1)
                [~,~, IY,IX]=subswapg(E,F);
                IY=IY*IRL31;
                IX=IRR31*IX;
                IYinv=[-IY(3,2:3) 1; eye(2) zeros(2,1)];
                
                aa=vs(ii)-bkc;
                bb=vs(ii)-bkc+2;
                IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
                IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
                IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
                
                T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
                S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
               
                T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
                S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
                T(aa+1,aa-1:aa)=zeros(1,2);
                S(aa+1,aa-1:aa)=zeros(1,2);
            elseif (s==0)
                zc=zc+1;
                zcl(zc)=vs(ii)-bkc;
            end
            bkc=bkc+1;
       elseif(T(vs(ii)-bkc-1,vs(ii)-bkc-2)~=0 && T(vs(ii)-bkc+1,vs(ii)-bkc)==0)
           E=T(vs(ii)-bkc-2:vs(ii)-bkc,vs(ii)-bkc-2:vs(ii)-bkc);
           F=S(vs(ii)-bkc-2:vs(ii)-bkc,vs(ii)-bkc-2:vs(ii)-bkc);
           %sprintf('m3')
           %T
           s=success(E,F);
           if(s==1)
               
               [~,~, IY,IX]=subswapg(E,F);
               IY=IY*IRL32;
               IX=IRR32*IX;
               IYinv=[-IY(2:3,3) eye(2); 1 zeros(1,2)];
               
               aa=vs(ii)-bkc-2;
               bb=vs(ii)-bkc;
               
               IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
               IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
               IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
               
               T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
               S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
           
               T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
               S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
               
               T(bb-1:bb,aa)=zeros(2,1);
               S(bb-1:bb,aa)=zeros(2,1);
           elseif (s==0)
               zc=zc+1;
               zcl(zc)=vs(ii)-bkc;
           end
           bkc=bkc+2;
       elseif(T(vs(ii)-bkc+1,vs(ii)-bkc)==0 && T(vs(ii)-bkc-1,vs(ii)-bkc-2)==0)
           E=T(vs(ii)-bkc:vs(ii)-bkc+1,vs(ii)-bkc:vs(ii)-bkc+1);
           F=S(vs(ii)-bkc:vs(ii)-bkc+1,vs(ii)-bkc:vs(ii)-bkc+1);
           %sprintf('m4')
           %T
           s=success(E,F);
           if(s==1)
               [~,~, IY,IX]=subswapg(E,F);
               
               IY=IY*IR2;
               IX=IR2*IX;
               IYinv=[-IY(2,2) 1;1 0];
               
               aa= vs(ii)-bkc;
               bb=vs(ii)-bkc+1;
               
               IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
               IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
               IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
               
               T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
               S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
          
               T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
               S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
               T(aa,aa-1)=0;
               S(aa,aa-1)=0;
           elseif (s==0)
               zc=zc+1;
               zcl(zc)=vs(ii)-bkc;
           end
           bkc=bkc+1;
       end
      end
      
      if(length(zcl)>1)
          if (T(zcl(1)+1,zcl(1)) == 0)
              vsl=length(vs);   
              vs=[vs(1:ii) zcl(1) vs(ii+1:vsl)];
          else
              vsl=length(vs);   
              vs=[vs(1:ii+1) zcl(1) zcl(1)+1 vs(ii+2:vsl)];
          end
       end
  
       elseif(vs(ii)==n)
           if(T(n,n-1)==0)
               if (T(n-1,n-2)==0) 
                   bkc=1;
                   zz=1;
                   E=T(n-1:n,n-1:n);
                   F=S(n-1:n,n-1:n);
                   %sprintf('m5')
                   %T
                   s=success(E,F);
                   if(s==1)
                       
                       [~,~, IY,IX]=subswapg(E,F);
                       IY=IY*IR2;
                       IX=IR2*IX;
                       IYinv=[-IY(2,2) 1;1 0];
                       
                       IYt(n-1:n,1:n)=IY*IYt(n-1:n,1:n);
                       IXt(1:n,n-1:n)=IXt(1:n,n-1:n)*IX;
                       IYtinv(1:n,n-1:n)=IYtinv(1:n,n-1:n)*IYinv;
                       
                       
                       T(n-1:n,n-1:n)=IY*T(n-1:n,n-1:n);
                       S(n-1:n,n-1:n)=IY*S(n-1:n,n-1:n);

                       T(1:n,n-1:n)=T(1:n,n-1:n)*IX;
                       S(1:n,n-1:n)=S(1:n,n-1:n)*IX;
                       T(n,n-1)=0;
                       S(n,n-1)=0;
                   elseif (s==0)
                       vs=[vs n];
                   end
               elseif(T(n-1,n-2)~=0)
                   bkc=2;
                   zz=2;
                   E=T(n-2:n,n-2:n);
                   F=S(n-2:n,n-2:n);
                   % sprintf('m6')
                   % T
                   s=success(E,F);
                   if(s==1)
                       
                       [~,~, IY,IX]=subswapg(E,F);
                       IY=IY*IRL32;
                       IX=IRR32*IX;
                       IYinv=[-IY(2:3,3) eye(2); 1 zeros(1,2)];
                       
                       IYt(n-2:n,1:n)=IY*IYt(n-2:n,1:n);
                       IXt(1:n,n-2:n)=IXt(1:n,n-2:n)*IX;
                       IYtinv(1:n,n-2:n)=IYtinv(1:n,n-2:n)*IYinv;
                       
                       T(n-2:n,n-2:n)=IY*T(n-2:n,n-2:n);
                       S(n-2:n,n-2:n)=IY*S(n-2:n,n-2:n);
           
                       T(1:n,n-2:n)=T(1:n,n-2:n)*IX;
                       S(1:n,n-2:n)=S(1:n,n-2:n)*IX;
                       T(n-1:n,n-2)=zeros(2,1);
                       S(n-1:n,n-2)=zeros(2,1);
                   elseif (s==0)
                       vs=[vs n];
                   end
               end
               
               for jj=tmbs+1:vs(ii)-cv-1-zz
                   zc=0;
                   zcl=zeros(1,1);
                   if(T(vs(ii)-bkc-1,vs(ii)-bkc-2)~=0 && T(vs(ii)-bkc+1,vs(ii)-bkc)~=0) 
                       E=T(vs(ii)-bkc:vs(ii)-bkc+3,vs(ii)-bkc:vs(ii)-bkc+3);
                       F=S(vs(ii)-bkc:vs(ii)-bkc+3,vs(ii)-bkc:vs(ii)-bkc+3);
                       % sprintf('m7')
                       % T
                       s=success(E,F);
                       if (s==1)
                           
                           [~,~, IY,IX]=subswapg(E,F);
                           IY=IY*IR4;
                           IX=IR4*IX;
                           IYinv=[-IY(3:4,3:4) eye(2); eye(2) zeros(2,2)];
                           
                           aa=vs(ii)-bkc;
                           bb=vs(ii)-bkc+3;
                           
                           IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
                           IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
                           IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
                           
                           T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
                           S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
             
                           T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
                           S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
                           
                           T(aa:aa+1,aa-2:aa-1)=zeros(2,2);
                           S(aa:aa+1,aa-2:aa-1)=zeros(2,2);
                       elseif (s==0)
                           zc=zc+1;
                           zcl(zc)=vs(ii)-bkc;
                       end
                       bkc=bkc+2;
                   elseif (T(vs(ii)-bkc-1,vs(ii)-bkc-2)==0 && T(vs(ii)-bkc+1,vs(ii)-bkc)~=0 )
                       E=T(vs(ii)-bkc-2:vs(ii)-bkc,vs(ii)-bkc-2:vs(ii)-bkc);
                       F=S(vs(ii)-bkc-2:vs(ii)-bkc,vs(ii)-bkc-2:vs(ii)-bkc);
                       % sprintf('m8')
                       % T
                       s=success(E,F);
                       if(s==1)
                           [~,~,IY,IX]=subswapg(E,F);
                           IY=IY*IRL31;
                           IX=IRR31*IX;
                           IYinv=[-IY(3,2:3) 1; eye(3) zeros(2,1)];
                           
                           aa=vs(ii)-bkc-2;
                           bb=vs(ii)-bkc;
                           
                           IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
                           IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
                           IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
                           
                           T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
                           S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
           
                           T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
                           S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
                           
                           T(bb+1,bb-1:bb)=zeros(1,2);
                           S(bb+1,bb-1:bb)=zeros(1,2);
                       elseif (s==0)
                           zc=zc+1;
                           zcl(zc)=vs(ii)-bkc;
                       end
                       bkc=bkc+1;
                   elseif (T(vs(ii)-bkc-1,vs(ii)-bkc-2)~=0 && T(vs(ii)-bkc+1,vs(ii)-bkc)==0)
                       E=T(vs(ii)-bkc-2:vs(ii)-bkc,vs(ii)-bkc-2:vs(ii)-bkc);
                       F=S(vs(ii)-bkc-2:vs(ii)-bkc,vs(ii)-bkc-2:vs(ii)-bkc);
                       sprintf('m9')
                       T
                       s=success(E,F);
                       if(s==1)
                           [~,~, IY,IX]=subswapg(E,F);
                           IY=IY*IRL32;
                           IX=IRR32*IX;
                           IYinv=[-IY(2:3,3) eye(2); 1 zeros(1,2)];
                           
                           aa=vs(ii)-bkc-2;
                           bb=vs(ii)-bkc;
                           
                           IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
                           IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
                           IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
               
                           T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
                           S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
           
                           T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
                           S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
               
                           T(bb-1:bb,aa)=zeros(2,1);
                           S(bb-1:bb,aa)=zeros(2,1);
                       elseif (s==0)
                           zc=zc+1;
                           zcl(zc)=vs(ii)-bkc;
                       end
                       bkc=bkc+2;
                   elseif( T(vs(ii)-bkc+1,vs(ii)-bkc)==0 && T(vs(ii)-bkc-1,vs(ii)-2-bkc)==0)
                       E=T(vs(ii)-bkc-1:vs(ii)-bkc,vs(ii)-bkc-1:vs(ii)-bkc);
                       F=S(vs(ii)-bkc-1:vs(ii)-bkc,vs(ii)-bkc-1:vs(ii)-bkc);
                       sprintf('m10')
                       T
                       s=success(E,F);
                       if(s==1)
                           [~,~, IY,IX]=subswapg(E,F);
                           IY=IY*IR2;
                           IX=IR2*IX;
                           IYinv=[-IY(2,2) 1;1 0];
                           
                           aa=vs(ii)-bkc-1;
                           bb=vs(ii)-bkc;
                           
                           IYt(aa:bb,1:n)=IY*IYt(aa:bb,1:n);
                           IXt(1:n,aa:bb)=IXt(1:n,aa:bb)*IX;
                           IYtinv(1:n,aa:bb)=IYtinv(1:n,aa:bb)*IYinv;
                           
                           T(aa:bb,aa:n)=IY*T(aa:bb,aa:n);
                           S(aa:bb,aa:n)=IY*S(aa:bb,aa:n);
         
                           T(1:bb,aa:bb)=T(1:bb,aa:bb)*IX;
                           S(1:bb,aa:bb)=S(1:bb,aa:bb)*IX;
                           
                           T(bb,aa)=0;
                           S(bb,aa)=0;
                       elseif (s==0)
                           zc=zc+1;
                           zcl(zc)=vs(ii)-bkc;
                       end
                       bkc=bkc+1;
                   end
               end
               if(length(zcl)>1)
                   if (T(zcl(1)+1,zcl(1)) == 0)
                       vsl=length(vs);   
                       vs=[vs(1:ii) zcl(1) vs(ii+1:vsl)];
                   else
                       vsl=length(vs);
                       vs=[vs(1:ii+1) zcl(1) zcl(1)+1 vs(ii+2:vsl)];
                   end
               end
               
           %%%    
           end
  end
  
  if(T(tmbs+2,tmbs+1)==0)
      tmbs=tmbs+1;
  elseif(T(tmbs+2,tmbs+1)~=0)
      tmbs=tmbs+2;
  end
  ii=pp;
 end
end