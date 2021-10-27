using LinearAlgebra 
using Statistics
using StatsBase


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

function boostrap_confint(X, y, bootstrap_samples)
    n, d = size(X)
    beta_arr = zeros(bootstrap_samples, d)

    for i in 1:bootstrap_samples
        sample_arr = sample(1:n,n) 
        sampled_X = X[sample_arr, :]
        sampled_y = y[sample_arr]
        beta_arr[i, :] = linear_regression(sampled_X, sampled_y)
    end 

    betas = linear_regression(X,y)
    confidence_intervals = zeros(n, 2)

    for i in 1:n
        beta_i = beta_arr[i, :]
        beta_std = std(beta_i)
        confidence_intervals[i, 1] = betas[i] - 1.96 * beta_std
        confidence_intervals[i, 2] = betas[i] + 1.96 * beta_std

    end 
    return confidence_intervals
end 


# Very Simple Stability Analysis 

A = [1 2 3; 4 1 6; 7 8 1]
Q, R = classical_gram_schmidt(A)

classical_error = check_stability(Q)
Q, R = modified_gram_schmidt(A)
modified_error = check_stability(Q)

errror_ratios = classical_error / modified_error
println("classical error $classical_error")
println("modified error $modified_error")
println("Error ratios: $errror_ratios")

