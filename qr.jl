using LinearAlgebra 

function classical_gram_schmidt(A)
    m, n = size(A)
    R = zeros(n, n)
    Q = zeros(m, n)
    for j in 1:n
        v_j = A[:, j]
        for i in 1:(j-1)
            R[i, j] = dot(Q[:, i], A[:, j])
            v_j = v_j - R[i, j] * Q[:, i]
        end 
        R[j, j] = norm(v_j)
        Q[:, j] = v_j / R[j, j]
    end
    return Q, R
end 

function modified_gram_schmidt(A)
    m, n = size(A)
    R = zeros(Float64, n, n)
    Q = zeros(Float64, m, n)

    V = zeros(Float64, n, n)
    for i in 1:n
        V[:, i] = A[:, i]
    end

    for i in 1:n
        R[i, i] = norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        for j in (i+1):n
            R[i, j] = dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
        end 
    end 
    return Q, R
end 


function check_stability(Q)
    m, n = size(Q)
    total_dot_error = 0 
    for i in 1:m
        for j in 1:n 
            if i != j
                total_dot_error += abs(dot(Q[:, i], Q[:, j]))
            end 
        end 
    end 
    return total_dot_error
end 


