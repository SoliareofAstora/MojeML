A=rand(randi(10));

zadane(A)

A(A<0.2)=-1;
A(A>=0.2)=1;
A

function res = zadane(a)
[maxi,maxj]= size(a);
    res = a;
    for i = 1:maxi
        for j = 1:maxj
            if a(i,j)<0.2
                res(i,j)=-1;
            else
                res(i,j)=1;
            end
        end
    end
end