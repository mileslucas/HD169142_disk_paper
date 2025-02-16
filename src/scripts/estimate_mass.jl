using Measurements
using Unitful
using UnitfulAstro
using CSV

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
    motion = 360u"°" / T
    return motion |> u"°/yr"
end



inner_peak = (5.390404040404034 ± 1.0873340926494737)u"°/yr"
inner_loc = 21.15u"AU"

outer_peak = (0.5874747474747468 ± 1.1999989116751846)u"°/yr"
outer_loc = (65±5)u"AU"


m1 = estimate_mass(inner_peak, inner_loc)
m2 = estimate_mass(outer_peak, outer_loc)

@info "" m1 m2

a1 = estimate_separation(inner_peak, 2u"Msun")
a2 = estimate_separation(outer_peak, 2u"Msun")

@info "" a1 a2

estimate_motion(outer_loc, 2u"Msun")