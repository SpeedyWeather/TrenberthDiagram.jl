# Trenberth energy-budget diagnostics: calculation from SpeedyWeather output
#
# This file defines `calc_trenberth_from_diagn`, which reads the surface and
# top-of-atmosphere energy fluxes out of a SpeedyWeather `diagn` object,
# converts them from spectral coefficients into a single global-mean (or
# global-summed) number per flux, and returns them as a Dict. This Dict is
# what the TrenberthCallback stores at every timestep.

"""
    calc_trenberth_from_diagn(diagn, model; SumFlag=false, return_grids=false)

Compute the set of fluxes needed for a Trenberth-style energy budget diagram
from the current model state.

Arguments:
- `diagn`        : the SpeedyWeather `DiagnosticVariables` object for the current step
- `model`        : the SpeedyWeather model object (needed for the sphere normalisation
                   and planet radius used to convert spectral coefficients to physical units)
- `SumFlag`      : if `true`, return globally-*summed* fluxes (Watts, i.e. flux times Earth's
                   surface area); if `false` (default), return globally-*averaged*
                   fluxes (W/m²)
- `return_grids` : if `true`, also return a second `Dict{Symbol}` containing a
                   *copy* of each of the 9 raw fluxes on the model's native
                   compute grid (e.g. `OctahedralGaussianGrid`), suitable for
                   later turning into a map. We deliberately do NOT convert to
                   a rectangular `FullGaussianGrid` here: that conversion
                   requires an extra spectral transform, and doing it for
                   every recorded step (most of which may never be looked at)
                   wastes both memory and time. Instead, that conversion is
                   done lazily, only for the one timestep actually being
                   displayed, inside `plot_trenberth_diagram`'s maps viewer.
                   Copying matters here regardless: SpeedyWeather reuses the
                   same underlying arrays on every timestep, so without
                   copying, every stored snapshot would end up pointing at
                   the same (constantly overwritten) array.

Returns a `Dict{Symbol, Float64}` with one entry per flux, plus three derived
quantities (`SW_net_sfc`, `LW_net_sfc`, `surface_net`). If `return_grids=true`,
returns a `(results, grids)` tuple instead.
"""
function calc_trenberth_from_diagn(diagn, model; SumFlag::Bool=false, return_grids::Bool=false)

    # Map each flux name to the matching field on `diagn.physics`. These are
    # grid-point space fields on the model's native compute grid (often a
    # reduced grid like OctahedralGaussianGrid).
    # NOTE: LHF (latent heat flux) is *not* stored directly: SpeedyWeather
    # only stores the surface *moisture* flux, so we convert moisture flux
    # -> latent heat flux further down by multiplying by the latent heat of
    # vaporisation of water.
    fields = Dict(
        :LHF   => diagn.physics.surface_humidity_flux,
        :SHF   => diagn.physics.sensible_heat_flux,
        :SSRU  => diagn.physics.surface_shortwave_up,
        :SLRU  => diagn.physics.surface_longwave_up,
        :SSRD  => diagn.physics.surface_shortwave_down,
        :SLRD  => diagn.physics.surface_longwave_down,
        :OSR   => diagn.physics.outgoing_shortwave,
        :OLR   => diagn.physics.outgoing_longwave,
        :albedo=> diagn.physics.albedo
    )

    area = 4π * model.planet.radius^2

    # Compute each flux. Wrapped in try/catch per-variable so that a single
    # missing/broken field doesn't stop every other diagnostic from being
    # computed: it just gets recorded as NaN and a warning is printed.
    results = Dict{Symbol, Float64}()
    grids = return_grids ? Dict{Symbol, Any}() : nothing
    for (k, f) in fields
        try
            # LHF needs the same moisture -> heat flux conversion whether
            # we're aggregating it to a scalar or keeping the full grid, so
            # apply it once up front and reuse `converted` for both.
            converted = k == :LHF ? f .* 2.5e6 : f   # 2.5e6 J/kg ~ latent heat of vaporisation

            # Grid -> spectral. The (0,0) coefficient, once normalised by
            # `norm_sphere`, is the global mean: this avoids having to
            # average over every grid point by hand.
            spec = transform(converted)
            mean_val = real(spec[1]) / model.spectral_transform.norm_sphere
            results[k] = SumFlag ? mean_val * area : mean_val

            if return_grids
                # Store the native-grid snapshot directly - no extra
                # transform needed, we already have `converted` in hand.
                # The reinterpolation to a rectangular FullGaussianGrid
                # (needed for plotting) happens later, on demand, only for
                # whichever single timestep is actually being viewed.
                grids[k] = copy(converted)
            end
        catch err
            @warn "calc_trenberth_from_diagn: could not compute $k: $err"
            results[k] = NaN
            return_grids && (grids[k] = nothing)
        end
    end

    # Derived quantities used by the Trenberth diagram itself:
    # net shortwave and net longwave at the surface, and the overall surface
    # energy balance (this should be close to zero at radiative equilibrium).
    results[:SW_net_sfc]  = results[:SSRD] - results[:SSRU]
    results[:LW_net_sfc]  = results[:SLRD] - results[:SLRU]
    results[:surface_net] = results[:SW_net_sfc] + results[:LW_net_sfc] - results[:LHF] - results[:SHF]

    return return_grids ? (results, grids) : results
end
