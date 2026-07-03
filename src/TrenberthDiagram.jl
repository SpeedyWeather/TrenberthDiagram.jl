module TrenberthDiagram

using SpeedyWeather, GLMakie, Observables, Dates, Statistics
using SpeedyWeatherInternals.Utils
using LowerTriangularArrays
using SpeedyTransforms

include("Trenberth_calc.jl")
include("Trenberth_callback.jl")
include("Trenberth_observables.jl")
include("Trenberth_diagram.jl")

export TrenberthCallback, calc_trenberth_from_diagn,
       build_trenberth_observables, build_flux_arrow_observables, get_solar_constant,
       plot_trenberth_diagram, show_var_names, to_dataframe

end 

