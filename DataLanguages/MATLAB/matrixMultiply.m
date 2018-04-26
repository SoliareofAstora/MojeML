A = [1 3 5; 2 4 7];
B = [-5 8 11; 3 9 21; 4 0 8];

result = multiply(A,B);
testA = A*B;
if isequal(result,testA)
    disp('ITS OK!');
end


A = [1 3 5; 2 4 7];
B = [-5 8 11; 4 0 8];

result = strangemultiply(A,B);
testB = B.*A;
if isequal(result,testB)
    disp('ITS OK!');
end


function res = multiply(a,b)
res = mtimes(a,b);
end

function res = strangemultiply(a,b)
if isequal(size(a),size(b))
    [maxi,maxj]= size(a);
    res = a;
    for i = 1:maxi
        for j = 1:maxj
            res(i,j) = a(i,j)*b(i,j);
        end
    end
end
end

