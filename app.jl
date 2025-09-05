module App

include("lib/DiscreteTimeSIModel.jl")  
using Turing, MCMCChains, GenieFramework, PlotlyBase, StippleLatex, StatsPlots, Random, Distributions, Statistics
using .DiscreteTimeSIModel
Random.seed!(14)

@genietools

@app begin
    # Define reactive variables 
    @in beta = 0.52
    @in gamma = 0.24
    @in noise_level = 0.3  
    @in S0_discrete = 990
    @in I0_discrete = 10
    @in R0_discrete = 0
    @in time_steps = 100
    @in delta_t = 0.1
    @in n_realizations = 50

    # Plot for ODE solutions
    @out solplot = []  
    @out solplot_layout = PlotlyBase.Layout(
        title="SI Model Simulation",
        xaxis_title="Time",
        yaxis_title="Population",
        template="plotly_white"
    )
    
    # Stochastic discrete model plots
    @out stochastic_plot = []
    @out stochastic_plot_layout = PlotlyBase.Layout(
        title="Stochastic Discrete SIR Model",
        xaxis_title="Time Steps",
        yaxis_title="Population",
        template="plotly_white"
    )
    
    @out ensemble_plot = []
    @out ensemble_plot_layout = PlotlyBase.Layout(
        title="Stochastic SIR Model Ensemble (Multiple Realizations)",
        xaxis_title="Time Steps", 
        yaxis_title="Population",
        template="plotly_white"
    )
    
    @out data_plot = []
    @out data_plot_layout = PlotlyBase.Layout(
        title="Synthetic Data",
        xaxis_title="Time",
        yaxis_title="Population",
        template="plotly_white"
    )
    @out beta_mean = 0.0
    @out gamma_mean = 0.0
    @out beta_std = 0.0
    @out gamma_std = 0.0
    @out beta_mcse = 0.0
    @out gamma_mcse = 0.0
    @out beta_ess_bulk = 0.0
    @out gamma_ess_bulk = 0.0
    @out beta_ess_tail = 0.0
    @out gamma_ess_tail = 0.0
    # @out summary_stats = 0

    @private u0 = [0.99, 0.01]  
    @private tspan = (0.0, 100.0)  
    @private t = 0.0:1.0:100.0  # Time points for solution
    @private true_p = [0.52, 0.24]

    
    @onchange beta, gamma, S0_discrete, I0_discrete, R0_discrete, time_steps, delta_t begin
        try
            # Single realization of stochastic model
            S_stoch, I_stoch, R_stoch = simulate_SIR_discrete_stochastic(
                S0_discrete, I0_discrete, R0_discrete, beta, gamma, time_steps; Δt=delta_t
            )
            
            time_points = 0:time_steps
            stochastic_plot = [
                PlotlyBase.scatter(x=time_points, y=S_stoch, mode="lines+markers", name="Susceptible (Stochastic)", line=Dict("color" => "blue")),
                PlotlyBase.scatter(x=time_points, y=I_stoch, mode="lines+markers", name="Infected (Stochastic)", line=Dict("color" => "red")),
                PlotlyBase.scatter(x=time_points, y=R_stoch, mode="lines+markers", name="Recovered (Stochastic)", line=Dict("color" => "green"))
            ]
        catch e
            println("Error in stochastic simulation: ", e)
        end
    end
    
    # Stochastic ensemble visualization
    @onchange n_realizations, beta, gamma, S0_discrete, I0_discrete, R0_discrete, time_steps, delta_t begin
        try
            # Multiple realizations for ensemble analysis
            S_ensemble, I_ensemble, R_ensemble = simulate_SIR_stochastic_ensemble(
                S0_discrete, I0_discrete, R0_discrete, beta, gamma, time_steps, n_realizations; Δt=delta_t
            )
            
            time_points = 0:time_steps
            
            # Calculate mean and percentiles for uncertainty bounds
            S_mean = mean(S_ensemble, dims=2)[:]
            I_mean = mean(I_ensemble, dims=2)[:]
            R_mean = mean(R_ensemble, dims=2)[:]
            
            S_lower = [quantile(S_ensemble[i,:], 0.05) for i in 1:eachindex(S_ensemble,1)]
            S_upper = [quantile(S_ensemble[i,:], 0.95) for i in 1:eachindex(S_ensemble,1)]
            I_lower = [quantile(I_ensemble[i,:], 0.05) for i in 1:eachindex(I_ensemble,1)]
            I_upper = [quantile(I_ensemble[i,:], 0.95) for i in 1:eachindex(I_ensemble,1)]
            R_lower = [quantile(R_ensemble[i,:], 0.05) for i in 1:eachindex(R_ensemble,1)]
            R_upper = [quantile(R_ensemble[i,:], 0.95) for i in 1:eachindex(R_ensemble,1)]

            ensemble_plot = [
                # Mean trajectories
                PlotlyBase.scatter(x=time_points, y=S_mean, mode="lines", name="Susceptible (Mean)", line=Dict("color" => "blue")),
                PlotlyBase.scatter(x=time_points, y=I_mean, mode="lines", name="Infected (Mean)", line=Dict("color" => "red")),
                PlotlyBase.scatter(x=time_points, y=R_mean, mode="lines", name="Recovered (Mean)", line=Dict("color" => "green")),
                
                # Uncertainty bands (90% CI)
                PlotlyBase.scatter(x=time_points, y=S_upper, fill="none", mode="lines", line=Dict("color" => "rgba(0,0,255,0)"), showlegend=false),
                PlotlyBase.scatter(x=time_points, y=S_lower, fill="tonexty", mode="lines", line=Dict("color" => "rgba(0,0,255,0)"), 
                    fillcolor="rgba(0,0,255,0.2)", name="Susceptible (90% CI)"),
                    
                PlotlyBase.scatter(x=time_points, y=I_upper, fill="none", mode="lines", line=Dict("color" => "rgba(255,0,0,0)"), showlegend=false),
                PlotlyBase.scatter(x=time_points, y=I_lower, fill="tonexty", mode="lines", line=Dict("color" => "rgba(255,0,0,0)"), 
                    fillcolor="rgba(255,0,0,0.2)", name="Infected (90% CI)"),
                    
                PlotlyBase.scatter(x=time_points, y=R_upper, fill="none", mode="lines", line=Dict("color" => "rgba(0,255,0,0)"), showlegend=false),
                PlotlyBase.scatter(x=time_points, y=R_lower, fill="tonexty", mode="lines", line=Dict("color" => "rgba(0,255,0,0)"), 
                    fillcolor="rgba(0,255,0,0.2)", name="Recovered (90% CI)")
            ]
        catch e
            println("Error in ensemble simulation: ", e)
        end
    end  
end

meta = Dict(
    "og:title" => "SI Model Simulation",
    "og:description" => "Real-time simulation of an SI epidemic model with adjustable parameters and Bayesian inference.",
    "og:image" => "/preview.jpg"
)

layout = DEFAULT_LAYOUT(meta=meta)
@page("/", "app.jl.html", layout)

end