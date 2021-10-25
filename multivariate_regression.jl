include("qr.jl")


function linear_regression(X, y) 
    Q, R = modified_gram_schmidt(X)
    f = transpose(Q) * y 
    betas = backsolve(R, f)
    return betas
end 


# Helper Functions 

function back_substitute(U, y)
    n, _ = size(U)
    x = zeros(n)
    for i in n:1 
        x[i] = y[i]
        for j in (i+1): n
            x[i] = x[i] - U[i, j] * x[j]
        end 
        x[i] = x[i] / U[i, i]
    end 
    return x 
end 

function back_solve(A, b)
    Q,R = modified_gram_schmidt(A)
    y = tranpose(Q) * b 
    x = back_substitute(R, y)
    return x
end 


