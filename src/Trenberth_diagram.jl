# Trenberth energy-budget diagram: interactive GLMakie visualisation
#
# This file turns the Observables produced by `build_trenberth_observables`
# (see trenberth_observables.jl) into an animated Trenberth-style diagram:
# a schematic atmosphere/surface split with arrows whose width scales with
# flux magnitude, a slider + play/pause to step through the simulation,
# a summary info panel, and click-to-popup time series for each arrow.
#
# Typical usage after a SpeedyWeather run:
#
#   obs   = build_trenberth_observables(cb)
#   arrows = build_flux_arrow_observables(obs.current_point)
#   solar  = get_solar_constant(model)
#   fig    = plot_trenberth_diagram(obs, arrows, solar)
#   display(fig)

using GLMakie, Observables, Dates

# Panel layout used by the "show all flux time series" figure and the
# individual popup plots: (field, title, color, row, col).
const TRENBERTH_PANELS = [
    (:SSRD,        "Surface SW Down",   :gold1,          1, 1),
    (:SSRU,        "Surface SW Up",     :yellow3,        1, 2),
    (:OSR,         "OSR",               :goldenrod1,     1, 3),
    (:OLR,         "OLR",               :darkred,        2, 1),
    (:SLRD,        "LW Down",           :firebrick1,     2, 2),
    (:SLRU,        "LW Up",             :red,            2, 3),
    (:SHF,         "Sensible Heat",     :deeppink1,      3, 1),
    (:LHF,         "Latent Heat",       :slateblue,      3, 2),
    (:surface_net, "Surface Net",       :seagreen,       3, 3),
    (:SW_net_sfc,  "Net SW Surface",    :darkgoldenrod,  4, 1),
    (:LW_net_sfc,  "Net LW Surface",    :orangered,      4, 2),
    (:albedo,      "Albedo",            :cornflowerblue, 4, 3),
]

# Which x-position on the diagram each flux arrow sits at, and its display
# title/colour: used to detect mouse clicks and open the matching popup.
const TRENBERTH_ARROW_TARGETS = [
    (-2.6, :SSRD, "Surface SW Down",  :gold1),
    (-1.9, :SSRU, "Surface SW Up",    :yellow3),
    (-1.3, :OSR,  "OSR",              :goldenrod1),
    ( 2.2, :OLR,  "OLR",              :darkred),
    ( 1.9, :SLRD, "LW Down",          :firebrick1),
    ( 2.9, :SLRU, "LW Up",            :red),
    ( 0.4, :SHF,  "Sensible Heat",    :deeppink1),
    ( 1.3, :LHF,  "Latent Heat",      :slateblue),
]

# Layout for the 9-raw-field maps viewer: (field, title, color, row, col) in
# a 3x3 grid. Colors match TRENBERTH_PANELS/TRENBERTH_ARROW_TARGETS above, so
# each flux has one consistent colour across the whole diagram, time series,
# and maps.
const TRENBERTH_MAP_PANELS = [
    (:SSRD,   "Surface SW Down",  :gold1,          1, 1),
    (:SSRU,   "Surface SW Up",    :yellow3,        1, 2),
    (:OSR,    "OSR",              :goldenrod1,     1, 3),
    (:OLR,    "OLR",              :darkred,        2, 1),
    (:SLRD,   "LW Down",          :firebrick1,     2, 2),
    (:SLRU,   "LW Up",            :red,            2, 3),
    (:SHF,    "Sensible Heat",    :deeppink1,      3, 1),
    (:LHF,    "Latent Heat",      :slateblue,      3, 2),
    (:albedo, "Albedo",           :cornflowerblue, 3, 3),
]

# draw_flux_arrow!: draws one flux as a vertical arrow whose shaft width
# scales (non-linearly) with the flux magnitude, with a live-updating label.
"""
    draw_flux_arrow!(ax, x, y_start, y_end, flux_obs, label, color; kwargs...)

Draw a single animated flux arrow on axis `ax`, running from `(x, y_start)`
to `(x, y_end)`. The arrow's width tracks `flux_obs` (an Observable{Float64}),
so it grows/shrinks automatically as the timestep changes.

Keyword arguments:
- `base_scale`   : flux magnitude that maps to a "typical" arrow width. Tune
                   this per-flux since fluxes have very different magnitudes
- `labelside`    : `:left` or `:right`, which side of the arrow the label sits on
- `label_offset` : (dx, dy) fine-tuning for label placement
"""
function draw_flux_arrow!(ax, x, y_start, y_end, flux_obs, label, color;
                          base_scale=300.0, labelside=:left, label_offset=(0.0, 0.0))

    # Arrow shaft width: normalise the flux by base_scale, apply a sub-linear
    # power (^0.7) so that small fluxes are still visible and huge fluxes
    # don't dominate the diagram, then clamp to a sensible pixel range.
    width_obs = map(flux_obs) do f
        normalized = abs(f) / base_scale
        width_factor = normalized^0.7
        clamp(width_factor * 35, 3.0, 40.0)
    end

    # +1 if the arrow points up the page, -1 if it points down.
    direction = sign(y_end - y_start)

    # Arrowhead half-width and height, both derived from the shaft width so
    # thicker arrows get proportionally bigger heads.
    head_width = map(width_obs) do w
        clamp(w * 0.018, 0.10, 0.30)
    end
    head_height_obs = map(width_obs) do w
        clamp(w * 0.008, 0.08, 0.15) * direction
    end

    # Stop the shaft short of y_end so the arrowhead sits on top of it
    # instead of the shaft poking through the head.
    y_shaft_end = lift(head_height_obs) do hh
        y_end - hh
    end

    # Draw a slightly-wider black outline first, then the coloured shaft on
    # top of it, to give the arrow a clean stroked edge.
    outline_width = map(width_obs) do w; w + 4 end
    lines!(ax, [x, x], lift(y_shaft_end) do yse; [y_start, yse] end,
           linewidth = outline_width, color = :black)
    lines!(ax, [x, x], lift(y_shaft_end) do yse; [y_start, yse] end,
           linewidth = width_obs, color = color)

    # Triangular arrowhead.
    arrow_head = lift(head_width, head_height_obs) do hw, hh
        [Point2f(x - hw, y_end - hh),
         Point2f(x + hw, y_end - hh),
         Point2f(x, y_end)]
    end
    poly!(ax, arrow_head, color = color, strokecolor = :black, strokewidth = 2)

    # Live-updating text label showing the current flux value.
    label_x = (labelside == :left ? x - 0.35 : x + 0.35) + label_offset[1]
    label_y  = (y_start + y_end) / 2 + label_offset[2]
    label_text = map(flux_obs) do f
        "$(label)\n$(round(Int, abs(f))) W/m²"
    end
    text!(ax, label_x, label_y,
          text = label_text, fontsize = 20,
          align = (:center, :center), color = :black, font = :bold)
end

# show_flux_popup: a small standalone window with the full time series for
# one flux, plus a marker showing where the main diagram currently is.
function show_flux_popup(open_windows, ts, ts_steps, n, time_idx, date_ticks, field::Symbol, title::String, color)
    # If a popup for this field is already open, just bring it to front
    # instead of opening a duplicate.
    if haskey(open_windows, field) && isopen(open_windows[field])
        GLMakie.focus(open_windows[field])
        return
    end

    ys = getfield(ts, field)

    popup_fig = Figure(size = (700, 400), backgroundcolor = :white, fontsize = 13)
    Label(popup_fig[0, 1], title,
          fontsize = 18, font = :bold, color = color, tellwidth = false)

    ax_p = Axis(popup_fig[1, 1],
        xlabel       = "Date",
        ylabel       = field == :albedo ? "–" : "W/m²",
        xlabelsize   = 12,   ylabelsize   = 12,
        xgridvisible = true, ygridvisible = true,
        xticks       = date_ticks, xticklabelrotation = 0.5,
    )
    lines!(ax_p, collect(ts_steps), ys, color = color, linewidth = 2.5)
    band!(ax_p, collect(ts_steps), fill(minimum(ys), n), ys, color = (color, 0.15))
    vlines!(ax_p, time_idx, color = :black, linewidth = 1.2, linestyle = :dash)
    scatter!(ax_p, lift(i -> [Point2f(i, ys[i])], time_idx),
             color = color, strokecolor = :black, strokewidth = 1.5, markersize = 11)

    # Live value readout in the corner, tracking the current timestep.
    val_text = lift(time_idx) do i
        field == :albedo ? "$(round(ys[i], digits=3))" : "$(round(Int, ys[i])) W/m²"
    end
    text!(ax_p, 0.02, 0.95, text = val_text, space = :relative,
          fontsize = 14, font = :bold, color = color, align = (:left, :top))

    open_windows[field] = display(GLMakie.Screen(), popup_fig)
end

# plot_trenberth_diagram: assembles the full interactive figure
"""
    plot_trenberth_diagram(obs, arrows, solar_constant)

Build the full interactive Trenberth diagram figure: the schematic
atmosphere/surface diagram with animated flux arrows, an info panel showing
the current numeric values, playback controls (slider/play/reset), a toggle
for a "show all flux time series" window, a toggle for a "show flux maps"
window (a 3x3 grid of spatial heatmaps for the 9 raw fluxes, only shown if
`obs.maps` is non-empty, i.e. the callback that produced `obs` recorded
map snapshots - each panel uses the YlOrRd colormap, a static colorrange
spanning the whole run so the colorbar doesn't rescale as you scrub through
time, and an outline colored to match that flux elsewhere in the diagram),
and click-to-popup time series for individual arrows. Time series axes
(both the "show all" window and individual popups) label their x-axis with
actual dates rather than raw timestep numbers.

Arguments:
- `obs`            : the NamedTuple returned by `build_trenberth_observables`
- `arrows`         : the NamedTuple returned by `build_flux_arrow_observables`
- `solar_constant` : Observable{Float64}, e.g. from `get_solar_constant(model)`

Returns the assembled `Figure`. Call `display(fig)` to actually show it, and
`start_animation()`/`stop_animation()` (returned via closures below, or call
`display(fig)` then use the on-figure Play button) to control playback.
"""
function plot_trenberth_diagram(obs, arrows, solar_constant)
    (; current_point, time_idx, nsteps, ts, maps, flux_series) = obs
    ts_steps = 1:nsteps
    n = nsteps

    # A handful of evenly-spaced (index, date-label) pairs, used as custom
    # xticks so time series plots show readable dates instead of raw
    # timestep numbers. We keep the line/scatter data itself positioned by
    # plain integer index (1:nsteps) - simplest and correct as long as
    # recorded steps are evenly spaced in time (e.g. the callback's
    # `schedule=Schedule(every=Day(1))`) - and just relabel the ticks.
    date_tick_idx = unique(round.(Int, range(1, nsteps, length=min(nsteps, 6))))
    date_tick_labels = [Dates.format(flux_series[i].datetime, "yyyy-mm-dd") for i in date_tick_idx]
    date_ticks = (date_tick_idx, date_tick_labels)

    fig = Figure(size = (1800, 1000), backgroundcolor = :white, fontsize = 40)

    ax = Axis(fig[1, 1],
        aspect          = DataAspect(),
        backgroundcolor = (:lightblue, 0.12),
        title           = "Energy Budget Diagram")
    hidedecorations!(ax)
    hidespines!(ax)
    xlims!(ax, -4.0, 4.0)
    ylims!(ax, -2.5, 3.2)

    # ── Atmosphere box ────────────────────────────────────────────────────
    poly!(ax, Point2f[(-3.8, -1.4), (3.8, -1.4), (3.8, 2.4), (-3.8, 2.4)],
          color = (:skyblue, 0.3), strokecolor = :steelblue, strokewidth = 3)
    text!(ax, 0, 1.9, text = "ATMOSPHERE", fontsize = 30,
          align = (:center, :center), font = :bold, color = :grey0)

    # ── Surface box ───────────────────────────────────────────────────────
    poly!(ax, Point2f[(-3.8, -2.4), (3.8, -2.4), (3.8, -1.4), (-3.8, -1.4)],
          color = (:seagreen, 0.35), strokecolor = :darkgreen, strokewidth = 3)
    text!(ax, 0, -1.8, text = "SURFACE", fontsize = 30,
          align = (:center, :center), font = :bold, color = :grey0)

    # ── Decorative clouds (two overlapping clusters of circles) ─────────────
    cloud_centers = [(1.9,1.2),(2.05,1.18),(1.75,1.18),(1.97,1.25),
                     (1.83,1.25),(1.9,1.33),(2.15,1.16),(1.65,1.16)]
    xs = first.(cloud_centers); ys_c = last.(cloud_centers)
    cloud_radii = [0.31,0.28,0.28,0.26,0.26,0.25,0.24,0.24]

    for (dx, dy, sf, jitter) in [
            (-0.60,  0.22, 0.98, ones(8)),
            ( 0.35,  0.30, 0.90, [1.00,0.95,0.95,1.08,1.08,0.97,0.96,0.94]),
        ]
        cx = xs .+ dx; cy = ys_c .+ dy; cr = cloud_radii .* sf .* jitter
        # Draw a black "outline" circle slightly larger than the fill circle
        # underneath it, to fake a stroked/soft edge.
        scatter!(ax, cx, cy, marker=:circle, markersize=cr.*300 .+ 10,
                 color=:black, strokewidth=0)
        scatter!(ax, cx, cy, marker=:circle, markersize=cr.*300,
                 color=:grey96, strokecolor=:transparent, strokewidth=0)
    end
    scatter!(ax, xs, ys_c, marker=:circle, markersize=cloud_radii.*300 .+ 11,
             color=:black, strokewidth=0)
    scatter!(ax, xs, ys_c, marker=:circle, markersize=cloud_radii.*300,
             color=:grey96, strokecolor=:transparent, strokewidth=0)

    # ── Flux arrows ───────────────────────────────────────────────────────
    draw_flux_arrow!(ax, -2.2,  2.9,  2.4, solar_constant, "Solar Constant", :khaki1,
                     base_scale=350, labelside=:left,  label_offset=(-0.2, 0.4))
    draw_flux_arrow!(ax, -1.9, -1.4, -0.4, arrows.SSRU_obs, "Surface SW\nUp", :yellow,
                     base_scale=250, labelside=:left,  label_offset=(0.9, 0.2))
    draw_flux_arrow!(ax, -1.3,  2.4,  2.9, arrows.OSR_obs,  "OSR",             :goldenrod1,
                     base_scale=250, labelside=:right, label_offset=(0.3, 0.15))
    draw_flux_arrow!(ax, -2.6,  0.5, -1.4, arrows.SSRD_obs, "Surface SW\ndown", :gold1,
                     base_scale=280, labelside=:left,  label_offset=(-0.3, 0))
    draw_flux_arrow!(ax,  2.2,  2.4,  2.9, arrows.OLR_obs,  "OLR",             :darkred,
                     base_scale=300, labelside=:right, label_offset=(0.3, 0))
    draw_flux_arrow!(ax,  1.9,  0.7, -1.4, arrows.SLRD_obs, "LW Down",         :firebrick1,
                     base_scale=400, labelside=:right, label_offset=(0.09, 0))
    draw_flux_arrow!(ax,  2.9, -1.4,  0.9, arrows.SLRU_obs, "LW Up",           :red,
                     base_scale=450, labelside=:right, label_offset=(0.16, 0))
    draw_flux_arrow!(ax,  0.4, -1.4,  0.0, arrows.SHF_obs,  "Sensible",        :deeppink1,
                     base_scale=120, labelside=:left,  label_offset=(-0.2, 0))
    draw_flux_arrow!(ax,  1.3, -1.4,  0.0, arrows.LHF_obs,  "Latent",          :slateblue,
                     base_scale=180, labelside=:left,  label_offset=(-0.01, 0))

    # ── Info panel: live numeric readout of every flux at the current step ──
    info_grid = GridLayout(fig[1, 2], tellwidth = true)
    Label(info_grid[1, 1], "Energy Balance", fontsize = 22, font = :bold,
          halign = :center, color = :steelblue)
    Label(info_grid[2, 1], "──────────────────────", fontsize = 15, halign = :center)

    vals_text = map(current_point) do p
        """
        Surface Net: $(round(Int, p.surface_net)) W/m²

        Albedo: $(round(p.albedo, digits=3))

        Shortwave:
        • Surface Shortwave Down: $(round(Int, p.SSRD)) W/m²
        • OSR: $(round(Int, p.OSR)) W/m²
        • Surface Shortwave Up: $(round(Int, p.SSRU)) W/m²
        • Net surface Shortwave: $(round(Int, p.SW_net_sfc)) W/m²

        Longwave:
        • OLR: $(round(Int, p.OLR)) W/m²
        • LW Up: $(round(Int, p.SLRU)) W/m²
        • LW Down: $(round(Int, p.SLRD)) W/m²
        • Net surface Longwave: $(round(Int, p.LW_net_sfc)) W/m²

        Other Fluxes:
        • LHF: $(round(Int, p.LHF)) W/m²
        • SHF: $(round(Int, p.SHF)) W/m²
        """
    end
    Label(info_grid[3, 1], vals_text, fontsize = 20, halign = :center,
          valign = :top, tellheight = false)

    time_text = map(current_point) do p
        "$(Dates.format(p.datetime, "yyyy-mm-dd HH:MM"))\nStep $(time_idx[])/$(nsteps)"
    end
    Label(info_grid[6, 1], time_text, fontsize = 20, halign = :center, color = :navyblue)
    Label(info_grid[7, 1], "Legend", fontsize = 20, font = :bold, halign = :center)
    Label(info_grid[8, 1], """
    Yellow: Shortwave
    Red: Longwave
    Pink: Sensible heat
    Blue: Latent heat

    Arrow width shows flux magnitude
    Click any arrow to see its time series
    """, fontsize = 20, halign = :center)

    # ── Playback controls: slider, play/pause, reset ────────────────────────
    control_grid = GridLayout(fig[2, 1:2])
    slider       = Slider(control_grid[1, 1], range = 1:nsteps, startvalue = 1)
    is_playing   = Observable(false)
    play_label   = Observable("Play")
    play_button  = Button(control_grid[1, 2], label = play_label,  tellwidth = false)
    reset_button = Button(control_grid[1, 3], label = "Reset",     tellwidth = false)

    # Keep the slider and the time_idx Observable in sync in both directions.
    on(slider.value) do v; time_idx[] = Int(round(v)) end
    on(time_idx)     do i; set_close_to!(slider, i)   end

    # Animation is run as an @async task so the UI stays responsive; we keep
    # a handle to it so it can be cleanly interrupted on stop/reset/close.
    animation_task = Ref{Union{Task, Nothing}}(nothing)

    function stop_animation()
        is_playing[] = false
        if animation_task[] !== nothing
            try schedule(animation_task[], InterruptException(), error=true) catch end
            animation_task[] = nothing
        end
    end

    function start_animation()
        stop_animation()
        is_playing[] = true
        animation_task[] = @async begin
            try
                while is_playing[]
                    # Loop back to the start once the last step is reached.
                    time_idx[] = time_idx[] >= nsteps ? 1 : time_idx[] + 1
                    sleep(0.06)
                end
            catch e
                e isa InterruptException || @warn "Animation error" exception=e
            end
        end
    end

    on(play_button.clicks) do _
        if is_playing[]
            stop_animation(); play_label[] = "Play"
        else
            start_animation(); play_label[] = "Pause"
        end
    end
    on(reset_button.clicks) do _
        stop_animation(); play_label[] = "Play"; time_idx[] = 1
    end

    # "Show all flux time series" / "Show flux maps" toggle buttons, side by side.
    button_row = GridLayout(fig[0, 1:2])

    all_fluxes_screen = Ref{Any}(nothing)
    all_btn_label     = Observable("Show All Flux Time Series")
    all_btn = Button(button_row[1, 1], label = all_btn_label,
                     tellwidth = false, fontsize = 16)

    on(all_btn.clicks) do _
        scr = all_fluxes_screen[]
        if !isnothing(scr) && isopen(scr)
            # Second click while open: close the window and reset the label.
            close(scr)
            all_fluxes_screen[] = nothing
            all_btn_label[] = "Show All Flux Time Series"
        else
            all_btn_label[] = "Hide All Flux Time Series"

            ts_fig = Figure(size = (1500, 1100), backgroundcolor = :white, fontsize = 13)
            Label(ts_fig[0, 1:3], "Flux Time Series",
                  fontsize = 24, font = :bold, color = :steelblue, tellwidth = false)

            # Only the bottom row shows date labels/ticks - the panels above
            # share the same x-axis, so repeating dates on every row is just
            # visual clutter. Computed from the layout rather than hardcoded,
            # in case TRENBERTH_PANELS ever gains/loses rows.
            bottom_row = maximum(p[end-1] for p in TRENBERTH_PANELS)

            for (field, title, color, row, col) in TRENBERTH_PANELS
                is_bottom = row == bottom_row
                ys  = getfield(ts, field)
                ax2 = Axis(ts_fig[row, col],
                    title              = title,       titlesize    = 14,
                    xlabel             = is_bottom ? "Date" : "",
                    ylabel             = field == :albedo ? "–" : "W/m²",
                    xlabelsize         = 11,          ylabelsize   = 11,
                    xgridvisible       = true,        ygridvisible = true,
                    xticks             = date_ticks,
                    xticklabelrotation = 0.5,
                    xticklabelsvisible = is_bottom,
                )
                lines!(ax2, collect(ts_steps), ys, color = color, linewidth = 2)
                band!(ax2, collect(ts_steps), fill(minimum(ys), n), ys, color = (color, 0.15))
                vlines!(ax2, time_idx, color = :black, linewidth = 1.2, linestyle = :dash)
                scatter!(ax2, lift(i -> [Point2f(i, ys[i])], time_idx),
                         color = color, strokecolor = :black,
                         strokewidth = 1.5, markersize = 10)
            end

            all_fluxes_screen[] = display(GLMakie.Screen(), ts_fig)
        end
    end

    # "Show flux maps" toggle button, only shown if maps were actually
    # recorded (build_trenberth_observables returns an empty Dict for `maps`
    # if the callback didn't store any, e.g. an older TrenberthCallback).
    if !isempty(maps)
        maps_screen     = Ref{Any}(nothing)
        maps_btn_label  = Observable("Show Flux Maps")
        maps_btn = Button(button_row[1, 2], label = maps_btn_label,
                          tellwidth = false, fontsize = 16)

        on(maps_btn.clicks) do _
            scr = maps_screen[]
            if !isnothing(scr) && isopen(scr)
                close(scr)
                maps_screen[] = nothing
                maps_btn_label[] = "Show Flux Maps"
            else
                maps_btn_label[] = "Hide Flux Maps"

                maps_fig = Figure(size = (1500, 1100), backgroundcolor = :white, fontsize = 13)
                Label(maps_fig[0, 1:3], "Flux Maps",
                      fontsize = 24, font = :bold, color = :steelblue, tellwidth = false)

                for (field, title, color, row, col) in TRENBERTH_MAP_PANELS
                    haskey(maps, field) || continue

                    # `maps[field]` holds native-grid Field snapshots (e.g.
                    # OctahedralGaussianGrid; see calc_trenberth_from_diagn's
                    # `return_grids` docstring in trenberth_calc.jl) - smaller
                    # to store than a rectangular grid, but not directly
                    # plottable. We convert every recorded step to a
                    # displayable matrix once, up front here (rather than
                    # lazily per-frame), for two reasons:
                    #   1) it lets us compute one correct, static color range
                    #      spanning the whole run, so the colorbar doesn't
                    #      rescale as you scrub through time (a static native-
                    #      grid range wouldn't quite be correct here, since
                    #      the spectral reinterpolation below can slightly
                    #      overshoot the native grid's min/max)
                    #   2) it makes scrubbing the slider afterwards instant,
                    #      since nothing needs recomputing per frame
                    # This costs nsteps small spectral transforms per field,
                    # paid once when you open this window, not on every
                    # animation frame.
                    #
                    # Per-step conversion:
                    #  1) transform(native_field) -> spectral coefficients
                    #  2) transform(spec) with no Grid keyword -> defaults to
                    #     FullGaussianGrid, i.e. a *rectangular* grid (reduced
                    #     grids like the native one can't be reshaped into a
                    #     matrix, which is what `heatmap` needs)
                    #  3) Matrix(...) unwraps that into a plain matrix -
                    #     SpeedyWeather's own documented way to do this, and
                    #     more reliable than letting heatmap!'s automatic
                    #     conversion recipe try to handle the Field wrapper
                    #     itself, which doesn't dispatch correctly through
                    #     `heatmap!(ax, ::Observable{Field})` on some
                    #     SpeedyWeather/Makie version combinations.
                    #  4) reverse(..., dims=2): SpeedyWeather unravels grid
                    #     data ordered north -> south, and Matrix()'s
                    #     column-major reshape puts that northernmost data in
                    #     column 1 - but heatmap! draws column 1 at the
                    #     *bottom* of the axis, so without this flip north
                    #     ends up at the bottom (maps render upside down).
                    frames = Vector{Matrix{Float32}}(undef, n)
                    conversion_ok = true
                    for i in 1:n
                        try
                            full_grid = transform(transform(maps[field][i]))
                            frames[i] = reverse(Matrix(full_grid), dims=2)
                        catch err
                            if conversion_ok   # only warn once per field, but always assign a placeholder below
                                @warn "Could not convert map snapshots for $field to matrices: $err"
                                conversion_ok = false
                            end
                            frames[i] = zeros(Float32, 2, 2)
                        end
                    end

                    # Static colorrange for this field, spanning every frame,
                    # so the colorbar doesn't shift as you scrub through time.
                    colorrange = if conversion_ok
                        lo = minimum(minimum, frames)
                        hi = maximum(maximum, frames)
                        lo == hi ? (lo - 1f0, hi + 1f0) : (lo, hi)   # avoid a degenerate (equal) range
                    else
                        Makie.automatic
                    end

                    grid_obs = map(i -> frames[i], time_idx)

                    # Each panel gets its own sub-layout (axis + slim colorbar)
                    # so the colorbar can't spill into the neighbouring panel's grid cell.
                    panel = GridLayout(maps_fig[row, col])
                    ax_m = Axis(panel[1, 1],
                        title      = title, titlesize = 14,
                        aspect     = DataAspect(),
                    )
                    hidedecorations!(ax_m)

                    # Colored outline matching this flux's colour elsewhere
                    # in the diagram. hidedecorations! above may hide spines
                    # depending on the Makie version, so re-enable them
                    # explicitly rather than relying on whatever the default
                    # happened to leave visible.
                    ax_m.topspinevisible = true; ax_m.bottomspinevisible = true
                    ax_m.leftspinevisible = true; ax_m.rightspinevisible = true
                    ax_m.topspinecolor = color; ax_m.bottomspinecolor = color
                    ax_m.leftspinecolor = color; ax_m.rightspinecolor = color
                    ax_m.spinewidth = 4

                    hm = heatmap!(ax_m, grid_obs, colormap = :YlOrRd, colorrange = colorrange)
                    Colorbar(panel[1, 2], hm, width = 10, label = field == :albedo ? "" : "W/m²")
                end

                maps_screen[] = display(GLMakie.Screen(), maps_fig)
            end
        end
    end

    # ── Click-to-popup: clicking near an arrow opens its individual time series ─
    open_windows = Dict{Symbol, Any}()

    on(events(ax).mousebutton) do event
        event.button == Mouse.left && event.action == Mouse.press || return
        mp = Makie.mouseposition(ax)
        mx, my = mp[1], mp[2]
        (-3.8 ≤ mx ≤ 3.8 && -2.4 ≤ my ≤ 2.9) || return  # ignore clicks outside the diagram
        for (ax_x, field, title, color) in TRENBERTH_ARROW_TARGETS
            if abs(mx - ax_x) < 0.35   # click tolerance either side of the arrow
                show_flux_popup(open_windows, ts, ts_steps, n, time_idx, date_ticks, field, title, color)
                return
            end
        end
    end

    # Start the animation playing immediately once the figure is shown.
    start_animation()
    play_label[] = "Pause"

    # A cleanup closure that tears down all background tasks and popup
    # windows in one call. Returned alongside `fig` since a Figure can't
    # hold arbitrary extra fields itself.
    cleanup = function ()
        stop_animation()
        for scr in values(open_windows)
            isopen(scr) && close(scr)
        end
        scr = all_fluxes_screen[]
        !isnothing(scr) && isopen(scr) && close(scr)
        if !isempty(maps)
            mscr = maps_screen[]
            !isnothing(mscr) && isopen(mscr) && close(mscr)
        end
    end

    return (fig=fig, cleanup=cleanup)
end
