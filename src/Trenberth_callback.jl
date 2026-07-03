# TrenberthCallback: a SpeedyWeather callback that records Trenberth energy
# budget diagnostics at every (scheduled) timestep of a simulation
#
# This file defines:
#   - TRENBERTH_LONGNAMES : human-readable names/units for each flux variable
#   - time_to_float        : small helper to convert DateTimes/Periods/Numbers to Float64
#   - TrenberthCallback    : the callback struct + constructor
#   - initialize!/callback!/finalize! : the three lifecycle hooks SpeedyWeather calls
#   - show_var_names/to_dataframe     : convenience helpers for inspecting results
#
# Usage pattern (see bottom of file for a worked example):
#   cb = TrenberthCallback(schedule=Schedule(every=Day(1)))
#   add!(model.callbacks, :trenberth => cb)
#   run!(simulation)
#   # cb.data now holds one Vector{Float64} per flux, one entry per recorded step

using Dates
using Statistics   # for `mean` in finalize!

# Long names / units for each flux, used for print-outs and DataFrame columns
const TRENBERTH_LONGNAMES = Dict(
    :LHF => "Surface latent heat flux (W/m²)",
    :SHF => "Surface sensible heat flux (W/m²)",
    :SSRU => "Surface shortwave up (W/m²)",
    :SLRU => "Surface longwave up (W/m²)",
    :SSRD => "Surface shortwave down (W/m²)",
    :SLRD => "Surface longwave down (W/m²)",
    :OSR => "Outgoing shortwave radiation (TOA) (W/m²)",
    :OLR => "Outgoing longwave radiation (TOA) (W/m²)",
    :albedo => "Surface albedo",
    :SW_net_sfc => "Surface net shortwave (W/m²)",
    :LW_net_sfc => "Surface net longwave (W/m²)",
    :surface_net => "Surface net energy (W/m²)"
)

# time_to_float: normalise DateTime / Period / Number into a plain Float64,
# in either seconds or days. Used wherever a time value needs to go into a
# plain numeric vector (e.g. for plotting against a numeric x-axis).
"""
    time_to_float(t; unit=:seconds)

Convert a `DateTime`, a `Dates.Period` (e.g. `Day`, `Hour`, `Minute`), or a
plain `Number` into a `Float64`, expressed in `:seconds` or `:days`.
"""
function time_to_float(t; unit::Symbol = :seconds)
    if t isa DateTime
        # Seconds since the Unix epoch (1970-01-01).
        secs = Dates.datetime2unix(t)
        return unit == :seconds ? Float64(secs) :
               unit == :days    ? Float64(secs / 86400.0) :
               error("unsupported unit: $unit")

    elseif t <: Dates.Period   # e.g. Day, Hour, Minute
        if unit == :days
            # Dates.value gives the raw integer count in the period's own units
            # (e.g. number of days for `Day`), so this is only correct if `t`
            # is already a Day-based period.
            return float(Dates.value(t))
        elseif unit == :seconds
            # Convert explicitly for the common period types, since each has
            # a different seconds-per-unit ratio.
            if t isa Day
                return float(Dates.value(t) * 86400)
            elseif t isa Hour
                return float(Dates.value(t) * 3600)
            elseif t isa Minute
                return float(Dates.value(t) * 60)
            else
                # Fallback for uncommon period types: round to whole days first.
                return float(Dates.value(Day(round(Int, Dates.value(t)))) * 86400)
            end
        else
            error("unsupported unit: $unit")
        end

    elseif t isa Number
        return float(t)
    else
        error("unsupported time type: $(typeof(t))")
    end
end

# The 9 raw flux fields that get map (spatial) snapshots stored, as opposed
# to the 3 derived fields (SW_net_sfc, LW_net_sfc, surface_net) which are
# simple scalar differences and aren't stored as maps.
const TRENBERTH_MAP_VARS = [:LHF, :SHF, :SSRU, :SLRU, :SSRD, :SLRD, :OSR, :OLR, :albedo]

# TrenberthCallback: struct definition
"""
    TrenberthCallback

A SpeedyWeather callback that records Trenberth energy-budget fluxes
(see `calc_trenberth_from_diagn`) at every scheduled timestep.

Fields:
- `timestep_counter` : how many times `callback!` has fired
- `data`             : `Symbol => Vector{Float64}`, one vector per flux variable (global mean/sum time series)
- `maps`             : `Symbol => Vector{Any}`, one vector of spatial grid snapshots per raw
                       flux (see `TRENBERTH_MAP_VARS`), one snapshot per recorded step. Each
                       snapshot is kept on the model's native (often reduced, e.g.
                       OctahedralGaussianGrid) compute grid rather than reinterpolated to a
                       rectangular grid, to keep storage as small as possible during the run;
                       the reinterpolation needed for plotting happens lazily in
                       `plot_trenberth_diagram`, only for the timestep actually being viewed.
                       This still stores a full 2D field at every recorded step, so it can use
                       a lot of memory for long runs or high resolutions.
- `times`            : elapsed seconds since the simulation started, one entry per recorded step
- `datetimes`        : the original `DateTime` for each recorded step
- `start_time`       : simulation start time as a Unix timestamp (Float64)
- `SumFlag`          : passed through to `calc_trenberth_from_diagn` (sum vs. mean)
- `var_longnames`    : human-readable names, used by `show_var_names`/`to_dataframe`
- `record_maps`      : if `false`, skip capturing/storing spatial map snapshots entirely
                       (both `maps` and the extra work in `calc_trenberth_from_diagn` to
                       produce them) - only the scalar time series (`data`) is recorded.
                       Turn this off if you don't need the maps viewer and want to save
                       the memory and per-step overhead. Defaults to `true`.
- `schedule`         : a SpeedyWeather `Schedule` controlling how often this callback runs
"""
Base.@kwdef mutable struct TrenberthCallback <: SpeedyWeather.AbstractCallback
    timestep_counter::Int = 0
    data::Dict{Symbol, Vector{Float64}} = Dict{Symbol, Vector{Float64}}()
    maps::Dict{Symbol, Vector{Any}} = Dict{Symbol, Vector{Any}}()
    times::Vector{Float64} = Float64[]          # elapsed seconds since start
    datetimes::Vector{DateTime} = DateTime[]    # original DateTime stamps
    start_time::Float64 = 0.0
    SumFlag::Bool = false
    var_longnames::Dict{Symbol,String} = TRENBERTH_LONGNAMES
    record_maps::Bool = true
    # IMPORTANT: Schedule() with no arguments is not guaranteed to mean "every
    # timestep". If you want the callback to run every step, pass an explicit
    # schedule, e.g. `schedule = Schedule(every=model.time_stepping.Δt_sec)`,
    # or `schedule = Schedule(every=Day(1))` for once-daily recording. Check
    # `cb.timestep_counter` after a run to confirm it actually fired more than once.
    schedule::Schedule = Schedule()
end

# Convenience constructor. Storage vectors always start empty and grow with
# `push!` in `initialize!`/`callback!`. we deliberately do NOT pre-allocate
# to a fixed `nsteps + 1` length here, because mixing pre-allocated `undef`
# slots with later `push!` calls silently produces a vector with garbage
# values in the middle and real values appended past the end. Dynamic growth
# is a little slower for very long runs, but for a diagnostic callback like
# this the difference is negligible and correctness matters more.
function TrenberthCallback(; vars = [:LHF,:SHF,:SSRU,:SLRU,:SSRD,:SLRD,:OSR,:OLR,:albedo,:SW_net_sfc,:LW_net_sfc,:surface_net],
                             SumFlag::Bool=false,
                             var_longnames::Dict{Symbol,String}=TRENBERTH_LONGNAMES,
                             record_maps::Bool=true,
                             schedule::Schedule=Schedule())
    d = Dict{Symbol, Vector{Float64}}()
    for v in vars
        d[v] = Float64[]
    end
    m = Dict{Symbol, Vector{Any}}()
    if record_maps
        for v in TRENBERTH_MAP_VARS
            m[v] = Any[]
        end
    end
    return TrenberthCallback(0, d, m, Float64[], DateTime[], 0.0, SumFlag, var_longnames, record_maps, schedule)
end

# Inspection helpers

"""Print each flux's short name next to its human-readable long name."""
function show_var_names(cb::TrenberthCallback)
    for (k, long) in cb.var_longnames
        println(string(k), " → ", long)
    end
    return nothing
end

"""
    to_dataframe(cb::TrenberthCallback)

Assemble the recorded fluxes into a `DataFrame` (one column per flux, using
its long name where available, plus a `time` column). Requires DataFrames.jl
to be installed. It is loaded lazily so it isn't a hard dependency of this file.
"""
function to_dataframe(cb::TrenberthCallback)
    try
        @eval using DataFrames
    catch
        error("DataFrames.jl not available. Install it with `using Pkg; Pkg.add(\"DataFrames\")`")
    end
    df = DataFrame(time = cb.datetimes)
    for (k, vec) in cb.data
        colname = get(cb.var_longnames, k, string(k))  # prefer the long name as the column header
        df[Symbol(colname)] = vec
    end
    return df
end

# SpeedyWeather lifecycle hook 1/3: initialize!: called once before the run
function SpeedyWeather.initialize!(cb::TrenberthCallback,
                                   progn::PrognosticVariables,
                                   diagn::DiagnosticVariables,
                                   model::AbstractModel)

    # Scheduled callbacks must initialise their own schedule against the clock.
    initialize!(cb.schedule, progn.clock)

    # Remember the simulation's start time (as a Unix timestamp) so later
    # steps can report elapsed time relative to it.
    cb.start_time = Dates.datetime2unix(progn.clock.time)

    # Record the very first entry: the initial condition, before any
    # timestepping has happened. We always push here (rather than writing to
    # a pre-allocated index) so storage stays simple and consistent with
    # what `callback!` does below.
    cb.timestep_counter = 1
    t0 = Dates.datetime2unix(progn.clock.time)
    dt0 = progn.clock.time

    if cb.record_maps
        res0, grids0 = calc_trenberth_from_diagn(diagn, model; SumFlag=cb.SumFlag, return_grids=true)
    else
        res0 = calc_trenberth_from_diagn(diagn, model; SumFlag=cb.SumFlag, return_grids=false)
    end
    for (k, v) in res0
        if !haskey(cb.data, k)
            cb.data[k] = Float64[]
        end
        push!(cb.data[k], v)
    end
    if cb.record_maps
        for k in TRENBERTH_MAP_VARS
            if !haskey(cb.maps, k)
                cb.maps[k] = Any[]
            end
            push!(cb.maps[k], grids0[k])
        end
    end

    push!(cb.times, t0 - cb.start_time)
    push!(cb.datetimes, dt0)
    return nothing
end

# SpeedyWeather lifecycle hook 2/3: callback!: called after every step
# (subject to the callback's schedule)
function SpeedyWeather.callback!(cb::TrenberthCallback,
                                 progn::PrognosticVariables,
                                 diagn::DiagnosticVariables,
                                 model::AbstractModel)

    # Scheduled callbacks should bail out immediately if this isn't a step
    # on which they're supposed to run.
    isscheduled(cb.schedule, progn.clock) || return nothing

    cb.timestep_counter += 1

    if cb.record_maps
        res, grids = calc_trenberth_from_diagn(diagn, model; SumFlag=cb.SumFlag, return_grids=true)
    else
        res = calc_trenberth_from_diagn(diagn, model; SumFlag=cb.SumFlag, return_grids=false)
    end

    # Append this step's values to each flux's vector. New flux keys that
    # weren't seen in initialize! get created on the fly.
    for (k, v) in res
        if !haskey(cb.data, k)
            cb.data[k] = [v]
        else
            push!(cb.data[k], v)
        end
    end

    # Append this step's map snapshot for each of the 9 raw fields.
    if cb.record_maps
        for k in TRENBERTH_MAP_VARS
            if !haskey(cb.maps, k)
                cb.maps[k] = Any[]
            end
            push!(cb.maps[k], grids[k])
        end
    end

    # Record elapsed time (seconds since simulation start) and the DateTime
    # for this step.
    current_time = Dates.datetime2unix(progn.clock.time)
    push!(cb.times, current_time - cb.start_time)
    push!(cb.datetimes, progn.clock.time)
    return nothing
end

# SpeedyWeather lifecycle hook 3/3: finalize!: called once after the run
"""
Print the simulation-mean value of every recorded flux. Left as a simple
summary rather than appending one more data point, since the fluxes at the
model's very last diagnostic step are already captured by the final
`callback!` call.
"""
function SpeedyWeather.finalize!(cb::TrenberthCallback,
                                   progn::PrognosticVariables,
                                   diagn::DiagnosticVariables,
                                   model::AbstractModel)
    if cb.timestep_counter <= 1
        @warn "TrenberthCallback only recorded the initial timestep (timestep_counter=$(cb.timestep_counter)). " *
              "This usually means cb.schedule never triggered during the run - check how `schedule` was set " *
              "when constructing TrenberthCallback."
    end

    println("\n=== Simulation Means ===")
    for (k, vec) in cb.data
        mean_val = mean(vec)
        println("Mean $k over simulation: $mean_val")
    end

    if !isempty(cb.maps)
        maps_bytes = sum(Base.summarysize(v) for v in values(cb.maps))
        println("\nStored map snapshots use approximately $(round(maps_bytes / 1024^2, digits=1)) MB " *
                "(one grid per recorded step, for $(length(TRENBERTH_MAP_VARS)) fields).")
    end
    return nothing
end

# Example usage: registering the callback with a SpeedyWeather model
# cb = TrenberthCallback(schedule=Schedule(every=Day(1)))  # be explicit about how often it runs
# add!(model.callbacks, :trenberth => cb)
# keys(model.callbacks)                      # should include :trenberth
# model.callbacks[:trenberth] === cb          # should be true
