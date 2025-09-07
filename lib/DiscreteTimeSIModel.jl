module DiscreteTimeSIModel

using Turing, PlotlyBase, MCMCChains, Random 

"""
Discrete-time SIR model simulation functions
"""

function simulate_SIR_discrete(S0, I0, R0, β, γ, steps)
    # S0, I0, R0: initial numbers of S, I, R
    # β: infection rate
    # γ: recovery rate
    # steps: number of time steps to simulate

    S = zeros(Float64, steps+1)
    I = zeros(Float64, steps+1)
    R = zeros(Float64, steps+1)

    S[1] = S0
    I[1] = I0
    R[1] = R0
    N = S0 + I0 + R0

    for t in 1:steps
        new_infections = round(Float64, β * S[t] * I[t] / N)
        new_recoveries = round(Float64, γ * I[t])

        new_infections = min(new_infections, S[t])
        new_recoveries = min(new_recoveries, I[t])

        S[t+1] = S[t] - new_infections
        I[t+1] = I[t] + new_infections - new_recoveries
        R[t+1] = R[t] + new_recoveries
    end

    return S, I, R
end

function simulate_SIR_discrete_stochastic(S0, I0, R0, β, γ, steps; Δt=0.01)
    S = zeros(Float64, steps+1)
    I = zeros(Float64, steps+1)
    R = zeros(Float64, steps+1)

    S[1] = S0
    I[1] = I0
    R[1] = R0
    N = S0 + I0 + R0

    for t in 1:steps
        new_infections = rand(Binomial(S[t], 1 - exp(-β * I[t] / N * Δt)))
        new_recoveries = rand(Binomial(I[t], 1 - exp(-γ * Δt)))

        new_infections = min(new_infections, S[t])
        new_recoveries = min(new_recoveries, I[t])

        S[t+1] = S[t] - new_infections
        I[t+1] = I[t] + new_infections - new_recoveries
        R[t+1] = R[t] + new_recoveries
    end

    return S, I, R
end

"""
Generate multiple stochastic realizations for ensemble analysis
"""
function simulate_SIR_stochastic_ensemble(S0, I0, R0, α, γ, steps, n_realizations; Δt=0.01)
    # Store results for all realizations
    S_ensemble = zeros(Float64, steps+1, n_realizations)
    I_ensemble = zeros(Float64, steps+1, n_realizations)
    R_ensemble = zeros(Float64, steps+1, n_realizations)
    
    for i in 1:n_realizations
        S, I, R = simulate_SIR_discrete_stochastic(S0, I0, R0, β, γ, steps; Δt=Δt)
        S_ensemble[:, i] = S
        I_ensemble[:, i] = I
        R_ensemble[:, i] = R
    end
    
    return S_ensemble, I_ensemble, R_ensemble
end

end # module SIModel