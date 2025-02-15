using AstroImages
using DataFrames
using QuadPol

datadir(args...) = abspath(joinpath(@__DIR__, "..", "data", args...))

name = "HD169142"
plx = 8.7053e-3  # " +- 0.0268e-3
dist_pc = 1 / plx
inclination = 12.5 # deg
pos_angle = 5 # deg, location of far side minor axis
stellar_mass = 2 # Msun


folders = [
    "20120726_NACO_H",
    "20140425_GPI_J",
    "20150503_IRDIS_J",
    "20150710_ZIMPOL_VBB",
    "20180715_ZIMPOL_VBB",
    "20210906_IRDIS_Ks",
    "20230707_VAMPIRES_MBI",
    "20240729_VAMPIRES_MBI",
]
wavelengths = [
    1.6,
    1.2,
    1.2,
    0.735,
    0.735,
    2.2,
    0.678,
    0.678

]

function QU_from_Qphi_Uphi(Qphi, Uphi=0)
    nx, ny = size(Qphi)
    xs = range(-nx/2 + 0.5, nx/2 - 0.5)
    ys = range(-ny/2 + 0.5, ny/2 - 0.5)
    sky_angles = atand.(-xs, ys')
    # define azimuthal stokes
    Q = @. -Qphi * cosd(2 * sky_angles)# - Uphi * sind(2 * sky_angles)
    U = @. -Qphi * sind(2 * sky_angles)# + Uphi * cosd(2 * sky_angles)
    return Q, U
end

function load_and_mask_data(folder; rin=15, rout=35)
    filename = datadir(folder, "diskmap", "$(folder)_HD169142_diskmap_r2_scaled.fits")
    image = AstroImage(filename)

    filename = datadir(folder, "diskmap", "$(folder)_HD169142_diskmap_radius.fits")
    radii_au = AstroImage(filename)

    mask = @. rin ≤ radii_au ≤ rout
    @. image[!mask] = 0
    return image
end

function measure_quadpol(image)
    Q, U = QU_from_Qphi_Uphi(image)
    stats = quadpol(Q, U; pa=pos_angle)
    return stats
end


rows = map(folders, wavelengths) do folder, wavelength
    image = load_and_mask_data(folder)
    stats = measure_quadpol(image)
    row = (;folder, wavelength, stats...)
    return row
end

table = DataFrame(rows)

using StatsPlots

sort!(table, "wavelength")

@df table plot(:wavelength, [:delta_090_270 :delta_045_315 :delta_135_225])
hline!([0], c=:black)

@df table plot(:wavelength, [:lambda_000_180 :lambda_045_135 :lambda_315_225])
hline!([0], c=:black)

@df table plot(:wavelength, [:lambda_a :lambda_b])
hline!([0], c=:black)

table[!, "Qd_Qphi"] = table[!, "Qd"] ./ table[!, "Qphi"]
table[!, "Ud_Qphi"] = table[!, "Ud"] ./ table[!, "Qphi"]

@df table plot(:wavelength, [:Qd_Qphi :Ud_Qphi])
hline!([0], c=:black)


@df table plot(:wavelength, [:Q000 :Q090 :Q180 :Q270] ./ :Qd)

@df table plot(:wavelength, abs.([:Q000 :Q090 :Q180 :Q270] ./ :Qphi))
@df table plot(:wavelength, abs.([:U045 :U135 :U225 :U315] ./ :Qphi))