# Building Observables from a TrenberthCallback's recorded data
#
# This file turns the raw `cb.data` (a Dict of plain Vector{Float64}, one per
# flux, filled in by TrenberthCallback during the run) into the Observable
# objects the animated diagram needs: a vector of per-timestep NamedTuples,
# a slider index, and a "current point" that automatically updates whenever
# either of those change.
#
# Run this *after* the SpeedyWeather simulation has finished and `cb` is
# fully populated.

using Observables, Dates

const TRENBERTH_VARS_ORDER = [:LHF, :SHF, :SSRU, :SLRU, :SSRD, :SLRD, :OSR, :OLR, :albedo,
                               :SW_net_sfc, :LW_net_sfc, :surface_net]

"""
    build_trenberth_observables(cb; skip_first=true)

Convert a `TrenberthCallback`'s recorded data into the Observables used by
the Trenberth diagram and time-series plots.

Arguments:
- `cb`          : a `TrenberthCallback` that has already been run (i.e. has
                  populated `cb.data` and `cb.datetimes`)
- `skip_first`  : if `true` (default), drop the very first recorded entry.
                  The first entry is the *initial condition*, captured before
                  the model has taken any timesteps. Its fluxes are often
                  unrepresentative (e.g. zero/uninitialised physics tendencies),
                  so by default we exclude it from the animated series.

Returns a NamedTuple with:
- `flux_series`  : Vector{NamedTuple}, one entry per (kept) timestep
- `history_obs`  : Observable wrapping `flux_series`
- `time_idx`     : Observable{Int}, the currently selected timestep (slider position)
- `current_point`: Observable{NamedTuple}, derived from `history_obs[time_idx[]]`
- `ts`           : NamedTuple of plain Vectors, one per flux, for static/full time-series plots
- `maps`         : Dict{Symbol, Vector}, one vector of grid snapshots per raw flux (see
                   `TRENBERTH_MAP_VARS` in trenberth_callback.jl), index-aligned with
                   `flux_series`/`time_idx` so `maps[:OLR][time_idx[]]` matches `current_point[]`
- `nsteps`       : number of timesteps kept (== length(flux_series))
"""
function build_trenberth_observables(cb; skip_first::Bool=true)

    # Find the common length across all recorded flux vectors. Vectors can
    # end up different lengths if e.g. a new flux key first appeared partway
    # through the run, so we take the minimum to guarantee every timestep we
    # keep has a value for every variable.
    n_total = minimum(length(cb.data[k]) for k in TRENBERTH_VARS_ORDER if haskey(cb.data, k))

    # Skip the first recorded timestep (the pre-integration initial state)
    # if requested: see docstring above for why this is the default. The
    # same start_idx is used below for cb.maps so the two stay index-aligned.
    start_idx = skip_first ? 2 : 1
    n = n_total - start_idx + 1

    # Build one NamedTuple per kept timestep. NamedTuples are used because
    # they're lightweight, immutable, and let downstream code access fields
    # by name (e.g. `current_point[].LHF`) instead of by Dict lookup.
    flux_series = Vector{NamedTuple}(undef, n)
    for (out_i, i) in enumerate(start_idx:n_total)
        flux_series[out_i] = (
            datetime    = cb.datetimes[i],
            LHF         = cb.data[:LHF][i],
            SHF         = cb.data[:SHF][i],
            SSRU        = cb.data[:SSRU][i],
            SLRU        = cb.data[:SLRU][i],
            SSRD        = cb.data[:SSRD][i],
            SLRD        = cb.data[:SLRD][i],
            OSR         = cb.data[:OSR][i],
            OLR         = cb.data[:OLR][i],
            albedo      = cb.data[:albedo][i],
            SW_net_sfc  = cb.data[:SW_net_sfc][i],
            LW_net_sfc  = cb.data[:LW_net_sfc][i],
            surface_net = cb.data[:surface_net][i],
        )
    end

    # Wrap the whole series in one Observable. Replacing history_obs[] and
    # notifying it will cascade to every derived Observable below.
    history_obs = Observable(flux_series)

    # The slider position: which timestep is currently displayed.
    time_idx = Observable(1)

    # Automatically re-picks the NamedTuple at `time_idx` whenever either
    # `history_obs` or `time_idx` changes.
    current_point = map((hist, idx) -> hist[idx], history_obs, time_idx)

    # Plain (non-Observable) Vectors for each flux, used by the static
    # "full time series" plots where we want to draw the whole line at once
    # rather than track a single moving point.
    ts = (
        LHF         = [p.LHF         for p in flux_series],
        SHF         = [p.SHF         for p in flux_series],
        SSRU        = [p.SSRU        for p in flux_series],
        SLRU        = [p.SLRU        for p in flux_series],
        SSRD        = [p.SSRD        for p in flux_series],
        SLRD        = [p.SLRD        for p in flux_series],
        OSR         = [p.OSR         for p in flux_series],
        OLR         = [p.OLR         for p in flux_series],
        albedo      = [p.albedo      for p in flux_series],
        SW_net_sfc  = [p.SW_net_sfc  for p in flux_series],
        LW_net_sfc  = [p.LW_net_sfc  for p in flux_series],
        surface_net = [p.surface_net for p in flux_series],
    )

    # Slice cb.maps with the exact same start_idx:n_total window used above,
    # so maps[:field][i] lines up with flux_series[i] / time_idx == i.
    # If cb has no `maps` field (e.g. an older callback instance) or it's
    # empty, `maps` comes back as an empty Dict and map-viewing is skipped
    # downstream.
    maps = Dict{Symbol, Vector{Any}}()
    if hasproperty(cb, :maps) && !isempty(cb.maps)
        for k in TRENBERTH_MAP_VARS
            haskey(cb.maps, k) || continue
            if length(cb.maps[k]) < n_total
                @warn "build_trenberth_observables: cb.maps[$k] is shorter than the flux time series; skipping maps for this field"
                continue
            end
            maps[k] = cb.maps[k][start_idx:n_total]
        end
    end

    return (flux_series=flux_series, history_obs=history_obs, time_idx=time_idx,
            current_point=current_point, ts=ts, maps=maps, nsteps=n)
end

# Per-flux scalar Observables, used to size/label the individual arrows on
# the diagram. Each one just tracks the matching field of `current_point`.
"""
    build_flux_arrow_observables(current_point)

Split `current_point` (a NamedTuple Observable) out into one scalar
Observable per flux, for convenience when wiring up individual diagram arrows.
"""
function build_flux_arrow_observables(current_point)
    return (
        LHF_obs         = map(p -> p.LHF,         current_point),
        SHF_obs         = map(p -> p.SHF,         current_point),
        SSRU_obs        = map(p -> p.SSRU,        current_point),
        SLRU_obs        = map(p -> p.SLRU,        current_point),
        SSRD_obs        = map(p -> p.SSRD,        current_point),
        SLRD_obs        = map(p -> p.SLRD,        current_point),
        OSR_obs         = map(p -> p.OSR,         current_point),
        OLR_obs         = map(p -> p.OLR,         current_point),
        SW_net_sfc_obs  = map(p -> p.SW_net_sfc,  current_point),
        albedo_obs      = map(p -> p.albedo,      current_point),
        surface_net_obs = map(p -> p.surface_net, current_point),
    )
end

# Solar constant: pulled from the model's parameters rather than the
# callback data, since it's a fixed model parameter rather than a diagnosed flux.
"""
    get_solar_constant(model)

Read the solar constant (W/m²) out of the model's parameters and wrap it in
an Observable, so it can be passed to `draw_flux_arrow!` alongside the other
(genuinely time-varying) flux Observables.
"""
function get_solar_constant(model)
    params = parameters(model)
    param_vec = vec(params)
    return Observable(param_vec.planet.solar_constant)
end
