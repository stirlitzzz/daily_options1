using Distributions, Random, StatsBase,LinearAlgebra


function initialize_parameters(K, D)
    μ = [randn(D) for _ in 1:K]        # Means of clusters
    σ² = [rand(Gamma(2.0, 2.0)) for _ in 1:K]  # Variances
    π = normalize(rand(Dirichlet(1:K)))  # Mixture weights
    return μ, σ², π
end

function sample_assignments(R, μ, σ², π, K)
    N = length(R)
    z = zeros(Int, N)
    
    for t in 1:N
        log_probs = [log(π[k]) + logpdf(Normal(μ[k][1], sqrt(σ²[k])),R[t]) for k in 1:K]
        probs=exp.(log_probs)
        w=Weights(probs)
        z[t] = sample(1:K, w)
    end
    
    return z
end

#=
function sample_assignments(R, μ, σ², π, K)
    N = length(R)
    z = zeros(Int, N)
    
    for t in 1:N
        probs = [π[k] * pdf(Normal(μ[k][1], sqrt(σ²[k])),R[t]) for k in 1:K]
        w=Weights(probs)
        z[t] = sample(1:K, w)
    end
    
    return z
end
=#

#=

function sample_assignments(R, μ, σ², π_values, K)
    N = length(R)
    z = zeros(Int, N)
    
    for t in 1:N
        # Compute log probabilities for each cluster
        log_probs = [log.(π_values[k]) - 0.5 * log.(2π * σ²[k]) - ((R[t] - μ[k][1])^2 / (2 * σ²[k])) for k in 1:K]
        
        # Stabilize log probabilities by subtracting the maximum log probability
        max_log_prob = maximum(log_probs)
        stabilized_probs = exp.(log_probs .- max_log_prob)
        
        # Normalize probabilities to sum to 1
        probs = stabilized_probs / sum(stabilized_probs)
        
        # Sample cluster assignment
        w = Weights(probs)
        z[t] = sample(1:K, w)
    end
    
    return z
end

=#

function update_parameters(R, z, K, μ_0, λ_0, α_0, β_0)
    μ = Vector{Float64}(undef, K)
    σ² = Vector{Float64}(undef, K)
    n_k = counts(z, 1:K)  # Correctly counts points per cluster
    
    for k in 1:K
        assigned_points = R[z .== k]  # Points assigned to cluster k
        
        if n_k[k] > 0
            # Update parameters for cluster k
            R̄_k = mean(assigned_points)
            μ_post = (λ_0 * μ_0 + n_k[k] * R̄_k) / (λ_0 + n_k[k])
            λ_post = λ_0 + n_k[k]
            α_post = α_0 + n_k[k] / 2
            β_post = β_0 + sum((assigned_points .- R̄_k).^2) / 2 +
                      λ_0 * n_k[k] * (R̄_k - μ_0)^2 / (2 * λ_post)
            
            # Sample from posterior
            #println("σ²=",σ²)
            #println("k=$k, μ_post=$μ_post, λ_post=$λ_post, α_post=$α_post, β_post=$β_post, R̄_k=$R̄_k")
            σ²[k] = rand(InverseGamma(α_post, β_post))
            μ[k] = rand(Normal(μ_post, sqrt(σ²[k] / λ_post)))
        else
            # Use prior if no points are assigned to cluster k
            μ[k] = rand(Normal(μ_0, sqrt(1 / λ_0)))
            σ²[k] = rand(InverseGamma(α_0, β_0))
        end
    end
    
    # Sample new mixture weights using Dirichlet posterior
    π = rand(Dirichlet(n_k .+ 1))
    
    return μ, σ², π #, μ_post, λ_post, α_post, β_post
end

function gibbs_sampling(R, K, iterations, verbose = false)
    N = length(R)
    D = 1  # 1D data
    
    # Prior parameters
    μ_0 = 0.0
    λ_0 = 1.0
    α_0 = 2.0
    β_0 = 2.0
    println("Prior: μ_0 = $μ_0, λ_0 = $λ_0, α_0 = $α_0, β_0 = $β_0")

    # Initialize parameters
    μ, σ², π = initialize_parameters(K, D)
    #μ=[0,5]
    #σ²=[0,5]
    z = sample_assignments(R, μ, σ², π, K)
    if verbose
        println("Initial: Means = $μ, Variances = $σ², Weights = $π")
        println("Initial assignments: $z")
    end

    # Perform Gibbs Sampling
    for iter in 1:iterations
        z = sample_assignments(R, μ, σ², π, K)
        μ, σ², π = update_parameters(R, z, K, μ_0, λ_0, α_0, β_0)
        
        if verbose
            println("Iteration $iter: Means = $μ, Variances = $σ², Weights = $π")
        end
        #println("u_0=$μ_0, λ_0=$λ_0, α_0=$α_0, β_0=$β_0")
    end

    return μ, σ², π, z
end
