using AstroImages
using ADI
using ProgressMeter
using Statistics
using BiweightStats
using Diskmap
using Unitful: °

datadir(args...) = abspath(joinpath(@__DIR__, "..", "data", args...))

cube = AstroImage(datadir("20230707", "20230707_HD169142_vampires_adi_cube.fits"))
ref_cube = AstroImage(datadir("20230707", "20230707_HD169141_vampires_adi_cube.fits"))
angles = AstroImage(datadir("20230707", "20230707_HD169142_vampires_adi_angles.fits"))
stokes_cube = AstroImage(datadir("20230707", "20230707_HD169142_vampires_stokes_cube.fits"))

crop_size = 300

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

@info "Setting up disk geometry"
diskmap = Diskmap.DiskMap(12.5°, 5°, r->zero(r); inner_rad=10, n_rad=200)
projection = Diskmap.project(diskmap, crop_size, crop_size; distance=113.5, pxscale=5.9e-3)


tot_intensity = similar(stokes_cube, crop_size, crop_size, size(stokes_cube, 4))
for wl_idx in axes(stokes_cube, 4)
    pol = crop(stokes_cube[:, :, 5, wl_idx], 300)
    dolp = 1
    g = 0
    pol_frac = @. hg(projection.scattering; pol_max=dolp, g)
    tot_intensity[:, :, wl_idx] = @.  pol / pol_frac
end

function rerotate(image, parangs)
    cube = similar(image, size(image)..., length(parangs))
    for i in axes(cube, 3)
        cube[:, :, i] .= derotate(image, -parangs[i])
    end
    return cube
end


res_pcrdi = similar(cube, crop_size, crop_size, size(cube, 3))
for wl_idx in axes(res_pcrdi, 3)
    @info "Starting reduction for wavelength $wl_idx"
    target = crop(cube[:, :, wl_idx, :], crop_size)
    ref = crop(ref_cube[:, :, wl_idx, :], crop_size)
    css_est = rerotate(crop(tot_intensity[:, :, wl_idx], crop_size), angles)
    css_sub = target .- css_est
    S = reconstruct(Framewise(LOCI(dist_threshold=0.9)), css_sub; ref=ref, angles)
    # S = reconstruct(PCA(20), target; ref=[ref ;;; css_sub], angles)
    res_pcrdi[:, :, wl_idx] = collapse!(target .- S, angles)
end
DS9.set(res_pcrdi)

save(datadir("20230707_HD169142_vampires_loci_pcrdi.fits"), AstroImage(res_pcrdi))
save(datadir("20230707_HD169142_vampires_loci_pcrdi_coll.fits"), AstroImage(collapse(res_pcrdi, method=mean)))

DS9.set(res_pcrdi)
DS9.set(mask_circle(collapse(res_pcrdi, method=mean), 105/5.9; fill=NaN))
DS9.set(mask_circle(collapse(res_pcrdi, method=mean), 105/5.9; fill=NaN) .* projection.radius.^2)
