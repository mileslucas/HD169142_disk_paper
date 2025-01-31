using ADI
using AstroImages
using BiweightStats
using Diskmap
using Distributions
using NPZ
using ProgressMeter
using Statistics
using AffineInvariantMCMC

datadir(args...) = abspath(joinpath(@__DIR__, "..", "data", args...))

stokes_cube = AstroImage(datadir("20240729", "20240729_HD169142_vampires_stokes_cube.fits"))

crop_size = 150

# precrop cubes
pdi_frames = stack(f -> crop(f, crop_size), eachslice(stokes_cube[:, :, 5, :], dims=3))

# for numerical stability, let's scale every 
# image to a number closer to unity
target_info = (;
    plx = 8.7053,
    plx_err = 0.0268,
    dist=114.87,
    dist_err=0.35,
)

### phase functions
power_law(r, A, gamma) = A * r^gamma
power_law(A, gamma) = r -> power_law(r, A, gamma)


function gaussian_ring(radius; r0, amp, sigma)
    return amp * exp(-((radius - r0) / (2 * sigma))^2)
end

function disk_model(params)
    incl = params[1]
    pa = params[2]
    radius = params[3]
    amplitude = params[4]
    width = params[5]

    @info "Generating Model" incl pa radius amplitude width

    diskmap = Diskmap.DiskMap(
        deg2rad(incl),
        deg2rad(pa),
        r -> zero(typeof(r));
        inner_rad=10,
        n_rad=100
    )
    projection = Diskmap.project(
        diskmap,
        crop_size,
        crop_size;
        distance=target_info.dist,
        pxscale=5.9e-3
    )

    return @. gaussian_ring(projection.radius; r0=radius, amp=amplitude, sigma=width)
end

function azimuthal_dip(theta; th0, amp, sigma)
        # Calculate the smallest angular difference accounting for wrap-around
        delta_theta = abs(theta - th0)
        delta_theta = min(delta_theta, 360 - delta_theta)
    return 1 - amp * exp(-(delta_theta / (2 * sigma))^2)
end

function disk_model_with_shadow(params)
    incl = params[1]
    pa = params[2]
    radius = params[3]
    amplitude = params[4]
    width = params[5]
    shadow_pa = params[6] # position angle
    shadow_depth = params[7] # percentage dip 
    shadow_width = params[8] # gaussian width in deg at disk radius


    @info "Generating Model" incl pa radius amplitude width shadow_pa shadow_depth shadow_width


    diskmap = Diskmap.DiskMap(
        deg2rad(incl),
        deg2rad(pa),
        r -> zero(typeof(r));
        inner_rad=10,
        n_rad=100
    )
    projection = Diskmap.project(
        diskmap,
        crop_size,
        crop_size;
        distance=target_info.dist,
        pxscale=5.9e-3
    )

    ring = @. gaussian_ring(projection.radius; r0=radius, amp=amplitude, sigma=width)
    dimming = @. azimuthal_dip(rad2deg(projection.azimuth); th0=shadow_pa, amp=shadow_depth, sigma=shadow_width)
    return projection, ring, dimming, dimming .* ring
end

projection, ring, dimming, full = disk_model_with_shadow(
    12.5,
    5,
)

# using AffineInvariantMCMC
# using Random

# numwalkers=20
# numsamples_perwalker = 1000
# burnin = 100

# priors = [
#     # Normal(12.5, 1), # incl
#     # Normal(5, 1), # pa
#     truncated(Normal(10, 5); lower=0), # A
#     truncated(Normal(1, 1); lower=0), # gamma
#     Uniform(-1, 1),
#     Uniform(-1, 1),
#     Uniform(-1, 1),
#     Uniform(-1, 1),
#     Uniform(0, 1),
#     Uniform(0, 1),
#     Uniform(0, 1),
#     Uniform(0, 1),
# ]

# X0 = stack(rand.(priors, numwalkers))'

# llhood = X -> -pcrdi_loss(cube, angles, ref_cube, pdi_frames, X)

# @info "Starting burnin"
# chain, llhoodvals = AffineInvariantMCMC.sample(
#     llhood,
#     numwalkers,
#     X0,
#     burnin,
#     1
# )
# @info "Performing rest of sampling"
# chain, llhoodvals = AffineInvariantMCMC.sample(
#     llhood,
#     numwalkers,
#     chain[:, :, end],
#     numsamples_perwalker,
#     1
# )
# @info "Flattening chains"
# flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(
#     chain,
#     llhoodvals
# )

# npzwrite(
#     datadir("20240729_chain.npz"),
#     Dict(
#         "chain" => flatchain,
#         "loglikes" => flatllhoodvals
#     )
# )

