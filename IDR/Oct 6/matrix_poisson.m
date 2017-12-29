%In the following code is constructed the matrix for Du=f. The domain is a 
%square 10*10 points and we have zero boundary conditons. So we have a 
%system with 8*8 unknowns and 8*8 unknowns. The calculation is fixed raw by raw.
%The matrix is U_unknown


clear all;
close all;
x=ones(1,8);
U_unknown=zeros(8,8);
counter_metavlitis= 1 ;
for i = 2:9 ;
    for j= 2:9 ;
        %stin eksisosi pou imaste
        U_unknown( ((i)-2)*8+((j)-1) , counter_metavlitis ) = 4 ;
        if( i-1 ~= 1 )% to apo pano i-1 , j
            U_unknown( ((i-1)-2)*8+((j)-1) ,counter_metavlitis ) = -1 ;
        end
        if( i+1 ~= 10 )%to apo kato i+1, j
            U_unknown( ((i+1)-2)*8+((j)-1) , counter_metavlitis  ) = -1 ;
            
        end
        if( j+1 ~= 10 )% to deksia i, j+1
            U_unknown ( ((i)-2)*8+((j+1)-1) , counter_metavlitis ) = -1 ;
        end
        if( j-1 ~= 1)% to aristera i , j-1
            U_unknown ( ((i)-2)*8+((j-1)-1) , counter_metavlitis ) = -1 ;
        end 
        counter_metavlitis = counter_metavlitis +1 ;
    end
end
spy(U_unknown)
transpose(x)*U_unknown*x

