include("qr.jl")

A = [1 2 3; 4 1 6; 7 8 1]
Q, R = classical_gram_schmidt(A)

classical_error = check_stability(Q)
Q, R = modified_gram_schmidt(A)
modified_error = check_stability(Q)

errror_ratios = classical_error / modified_error
println("classical error $classical_error")
println("modified error $modified_error")
println("Error ratios: $errror_ratios")

