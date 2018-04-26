a = [2,3,1,1,1];
[suma,iloczyn] = suma_iloczyn(a)


function [suma,iloczyn] = suma_iloczyn(x) 
    suma = sum(x);
    iloczyn=1.;
    for i=1:length(x)
        iloczyn = iloczyn * x(i);
    end
end




