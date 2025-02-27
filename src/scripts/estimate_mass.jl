using Measurements
using Unitful
using UnitfulAstro
using CSV, DataFrames

Mstar = (1.79±0.17)u"Msun"

cc_values = CSV.read("/Users/mileslucas/research/papers/HD169142_disk_paper/src/data/cross_correlation_peaks.csv", DataFrame, header=false)
peak_values = CSV.read("/Users/mileslucas/research/papers/HD169142_disk_paper/src/data/radial_profile_peaks.csv", DataFrame, header=false)

inner_motion = (cc_values[1, :Column2] ± cc_values[1, :Column3])u"°/yr"
outer_motion = (cc_values[2, :Column2] ± cc_values[2, :Column3])u"°/yr"

inner_peak = (peak_values[1, :Column3] ± peak_values[1, :Column4])u"AU"
outer_peak = (peak_values[2, :Column3] ± peak_values[2, :Column4])u"AU"


function estimate_mass(motion, separation)
    T = 360u"°" / motion
    M = 4π^2 / Unitful.G * separation^3 / T^2
    return uconvert(u"Msun", M)
end


function estimate_separation(motion, mass)
    T = 360u"°" / motion
    a = cbrt(mass * T^2 * Unitful.G / (4π^2))
    return a |> u"AU"
end


function estimate_motion(separation, mass)
    T = sqrt(4π^2 / Unitful.G * separation^3 / mass)
    motion = -360u"°" / T
    return motion |> u"°/yr"
end

function ttest(a, b)
    diff = abs(Measurements.value(a) - Measurements.value(b))
    err = hypot(Measurements.uncertainty(a), Measurements.uncertainty(b))
    t = diff / err
    return diff, err, t
end


lines = String[]
m1 = estimate_mass(inner_motion, inner_peak)
push!(lines, "estimated mass from object at $(inner_peak) with $(inner_motion) motion: $m1")
m2 = estimate_mass(outer_motion, outer_peak)
push!(lines, "estimated mass from object at $(outer_peak) with $(outer_motion) motion: $m2")


a1 = estimate_separation(inner_motion, Mstar)
push!(lines, "estimated separation from object with $(inner_motion) motion around a $(Mstar) star: $a1")
a2 = estimate_separation(outer_motion, Mstar)
push!(lines, "estimated separation from object with $(outer_motion) motion around a $(Mstar) star: $a2")


kep_21 = estimate_motion(inner_peak, Mstar)
push!(lines, "expected motion around a $(Mstar) star at $(inner_peak): $(kep_21)")
kep_66 = estimate_motion(outer_peak, Mstar)
push!(lines, "expected motion around a $(Mstar) star at $(outer_peak): $(kep_66)")
kep_22 = estimate_motion((22.7±4.7)u"AU", Mstar)
push!(lines, "expected motion around a $(Mstar) star at $((22.7±4.7)u"AU"): $(kep_22)")
kep_37 = estimate_motion((37.2±1.5)u"AU", Mstar)
push!(lines, "expected motion around a $(Mstar) star at $((37.2±1.5)u"AU"): $(kep_37)")

zscore_21 = ttest(kep_21, inner_motion)
push!(lines, "zscore for inner ring motion vs Kep: $zscore_21")
zscore_37 = ttest(kep_37, inner_motion)
push!(lines, "zscore for inner ring motion vs 37au: $zscore_37")
zscore_0 = abs(Measurements.value(inner_motion) / Measurements.uncertainty(inner_motion))
push!(lines, "zscore for inner ring motion vs none: $zscore_0")

zscore_66 = ttest(kep_66, outer_motion)
push!(lines, "zscore for outer ring motion vs Kep: $zscore_66")
zscore_37 = ttest(kep_37, outer_motion)
push!(lines, "zscore for outer ring motion vs 37au: $zscore_37")
zscore_0 = abs(Measurements.value(outer_motion) / Measurements.uncertainty(outer_motion))
push!(lines, "zscore for outer ring motion vs none: $zscore_0")

print(join(lines, "\n"))