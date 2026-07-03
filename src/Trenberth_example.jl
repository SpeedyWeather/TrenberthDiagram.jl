# End to end usage example for the Trenberth diagram package
#
# Load order matters: trenberth_calc.jl defines calc_trenberth_from_diagn,
# which trenberth_callback.jl depends on.
#include("Trenberth_calc.jl")
#include("Trenberth_callback.jl")
#include("Trenberth_observables.jl")
#include("Trenberth_diagram.jl")

using TrenberthDiagram

using Pkg
using SpeedyWeather, GLMakie, Statistics
using SpeedyWeatherInternals.Utils
using LowerTriangularArrays
using SpeedyTransforms

# 1) set up the model
spectral_grid = SpectralGrid(trunc=31, nlayers=8)

model = PrimitiveWetModel(spectral_grid; shortwave_radiation=OneBandShortwave(spectral_grid, clouds = DiagnosticClouds(spectral_grid; use_stratocumulus=true)))

# 2) attach the callback before initializing the simulation, since
# initialize!(model) is what calls SpeedyWeather.initialize! on cb.
# Pass an explicit schedule - Schedule() with no arguments is not guaranteed
# to mean "every timestep", and relying on its default is what caused the
# callback to only ever record the initial (pre-run) state previously.
#
# record_maps=true (the default) captures a full spatial snapshot of all 9
# raw fluxes at every recorded step, enabling the "Show Flux Maps" viewer
# below - this uses more memory than the scalar time series alone. Set
# record_maps=false if you only want the time series / arrow diagram and
# want to skip the map storage and its per-step overhead entirely, e.g.:
#   cb = TrenberthCallback(schedule=Schedule(every=Day(1)), record_maps=false)
cb = TrenberthCallback(schedule=Schedule(every=Hour(12)))
add!(model.callbacks, :trenberth => cb)

# 3) initialize and run
sim = initialize!(model)   # this will call SpeedyWeather.initialize! on cb
run!(sim, period=Day(360))  # or your usual run invocation

# --- sanity checks -----------------------------------------------------
# If cb.timestep_counter is only 1 after the run, callback! never fired -
# check the `schedule` passed to TrenberthCallback above.
println("\ncb.timestep_counter = $(cb.timestep_counter) (should be roughly the number of days simulated)")
println("cb.data[:OLR] has $(length(cb.data[:OLR])) recorded values")
if cb.record_maps
    println("cb.maps[:OLR] has $(length(cb.maps[:OLR])) recorded map snapshots")
else
    println("cb.record_maps = false, so no map snapshots were recorded")
end
# finalize! (called automatically at the end of run!) already printed the
# per-flux simulation means, and (if record_maps=true) the approximate
# memory used by cb.maps, above.

# 4) build the Observables (skips the initial/pre-run timestep by default)
obs    = build_trenberth_observables(cb)
arrows = build_flux_arrow_observables(obs.current_point)
solar  = get_solar_constant(model)

println("\nobs.nsteps = $(obs.nsteps) (timesteps available to animate, after skipping the first)")
println("obs.maps has entries for: $(collect(keys(obs.maps)))")

# 5) build and display the interactive diagram
fig, cleanup = plot_trenberth_diagram(obs, arrows, solar)
display(fig)

println("\nDiagram is up.")
println("Click 'Show All Flux Time Series' for the full time series (dates now")
println("label only the bottom row of panels, to cut down on clutter).")
if !isempty(obs.maps)
    println("Click 'Show Flux Maps' for the 9 animated spatial maps - each uses the")
    println("YlOrRd colormap, a colorbar range that's fixed for the whole run (so it")
    println("doesn't rescale as you scrub through time), and an outline colored to")
    println("match that flux elsewhere in the diagram. Both windows stay in sync")
    println("with the main slider.")
else
    println("Maps were disabled for this run (record_maps=false), so no 'Show Flux")
    println("Maps' button is shown.")
end

# When you're done (e.g. closing the app), call cleanup() to stop the
# animation task and close any open popup windows.
