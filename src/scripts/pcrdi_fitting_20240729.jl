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

cube = AstroImage(datadir("20240729", "20240729_HD169142_vampires_adi_cube.fits"))
ref_cube = AstroImage(datadir("20240729", "20240729_HD317501_vampires_adi_cube.fits"))
angles = AstroImage(datadir("20240729", "20240729_HD169142_vampires_adi_angles.fits"))
stokes_cube = AstroImage(datadir("20240729", "20240729_HD169142_vampires_stokes_cube.fits"))

crop_size = 300

# precrop cubes
cube = stack(f -> crop(f, crop_size), eachslice(cube, dims=3))
ref_cube = stack(f -> crop(f, crop_size), eachslice(ref_cube, dims=3))
pdi_frames = stack(f -> crop(f, crop_size), eachslice(stokes_cube[:, :, 5, :], dims=3))

# for numerical stability, let's scale every 
# image to a number closer to unity
scale_factor = 10^(round(log10(mean(cube))))
cube ./= scale_factor
ref_cube ./= scale_factor
pdi_frames ./= scale_factor

target_info = (;
    plx = 8.7053,
    plx_err = 0.0268,
    dist=114.87,
    dist_err=0.35,
)

### phase functions

function rayleigh(scat_angle; pol_max=1)
    alpha = cos(scat_angle)
    return pol_max * (1 - alpha^2) / (alpha^2 + 1)
end

function hg(scat_angle; pol_max=1, g=0)
    ray = rayleigh(scat_angle; pol_max=pol_max)
    hg_fac = (1 - g^2) / sqrt((1 + g^2 - 2*g*cos(scat_angle))^3)
    return hg_fac * ray
end

function rerotate!(output, image, parangs)
    for i in axes(output, 3)
        output[:, :, i] .= derotate(image, -parangs[i])
    end
    return output
end
function rerotate!(image, parangs)
    output = similar(image, size(image)..., length(parangs))
    return rerotate!(output, image, parangs)
end
power_law(r, A, gamma) = A * r^gamma
power_law(A, gamma) = r -> power_law(r, A, gamma)

function pcrdi_loss(cube, angles, ref_cube, pdi_frames, params)
    # incl = params[1]
    # pa = params[2]
    power_law_A = params[1]
    power_law_gamma = params[2]
    gs = @view params[3:6]
    dolps = @view params[7:10]

    @info "Generating PCRDI Model" incl=12.5 pa=5 power_law_A power_law_gamma gs dolps

    diskmap = Diskmap.DiskMap(
        deg2rad(12.5),
        deg2rad(5),
        power_law(power_law_A, power_law_gamma);
        inner_rad=10,
        n_rad=100
    )
    projection = Diskmap.project(
        diskmap,
        crop_size,
        crop_size;
        distance=114.87,
        pxscale=5.9e-3
    )
    total_intensity = similar(projection.scattering)
    css_est = similar(cube, size(cube, 1), size(cube, 2), size(cube, 3))
    css_sub = similar(css_est)
    loss = zero(eltype(cube))
    @views for wl_idx in axes(cube, 4)
        total_intensity .= pdi_frames[wl_idx] ./ hg.(projection.scattering; pol_max=dolps[wl_idx], g=gs[wl_idx])
        @. total_intensity[isnan(total_intensity)] = 0
        rerotate!(css_est, total_intensity, angles)
        css_sub .= cube[:, :, :, wl_idx] .- css_est
        resid_frame = process(Framewise(LOCI(dist_threshold=0.9)), css_sub, angles; ref=ref_cube[:, :, :, wl_idx])
        loss += sum(s -> s^2, resid_frame)
    end
    return convert(Float64, loss)
end

using AffineInvariantMCMC
using Random

numwalkers=20
numsamples_perwalker = 1000
burnin = 100

priors = [
    # Normal(12.5, 1), # incl
    # Normal(5, 1), # pa
    truncated(Normal(10, 5); lower=0), # A
    truncated(Normal(1, 1); lower=0), # gamma
    Uniform(-1, 1),
    Uniform(-1, 1),
    Uniform(-1, 1),
    Uniform(-1, 1),
    Uniform(0, 1),
    Uniform(0, 1),
    Uniform(0, 1),
    Uniform(0, 1),
]

X0 = stack(rand.(priors, numwalkers))'

llhood = X -> -pcrdi_loss(cube, angles, ref_cube, pdi_frames, X)

@info "Starting burnin"
chain, llhoodvals = AffineInvariantMCMC.sample(
    llhood,
    numwalkers,
    X0,
    burnin,
    1
)
@info "Performing rest of sampling"
chain, llhoodvals = AffineInvariantMCMC.sample(
    llhood,
    numwalkers,
    chain[:, :, end],
    numsamples_perwalker,
    1
)
@info "Flattening chains"
flatchain, flatllhoodvals = AffineInvariantMCMC.flattenmcmcarray(
    chain,
    llhoodvals
)

npzwrite(
    datadir("20240729_chain.npz"),
    Dict(
        "chain" => flatchain,
        "loglikes" => flatllhoodvals
    )
)

