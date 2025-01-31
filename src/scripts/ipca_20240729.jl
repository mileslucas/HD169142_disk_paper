using AstroImages
using ADI
using ProgressMeter
using Statistics
using BiweightStats

datadir(args...) = abspath(joinpath(@__DIR__, "..", "data", args...))

cube = AstroImage(datadir("20240729", "20240729_HD169142_vampires_adi_cube.fits"))
ref_cube = AstroImage(datadir("20240729", "20240729_HD317501_vampires_adi_cube.fits"))
angles = AstroImage(datadir("20240729", "20240729_HD169142_vampires_adi_angles.fits"))
tot_intensity = AstroImage(datadir("20240729", "diskmap", "20240729_HD169142_vampires_total_intensity.fits"))
# IPCA consists of iteratively performing PCA on the science data  
# cube Y while subtracting the previously estimated disk signal
# d ̄i  at each step

function rerotate(image, parangs)
    cube = similar(image, size(image)..., length(parangs))
    for i in axes(cube, 3)
        cube[:, :, i] .= derotate(image, -parangs[i])
    end
    return cube
end

# the process starts with d0=-
function ipca(target, angles; ncomps=[1], iter_per_comp=[10])
    d = zeros(eltype(target), size(target, 1), size(target, 2))
    # for a given rank q, one iterative step is detailed as follows
    @showprogress "ncomps" foreach(ncomps, iter_per_comp) do ncomp, max_iter
        @showprogress "iterations" for i in 1:max_iter
            tmp_targ = target .- rerotate(abs.(d), angles) 
            S = reconstruct(PCA(ncomp), tmp_targ)
            d = collapse!(target .- S, angles)
        end
    end
    return d
end


# the process starts with d0=-
function ipca(target, angles, ref; ncomps=[1], iter_per_comp=[10])
    d = zeros(eltype(target), size(target, 1), size(target, 2))
    # for a given rank q, one iterative step is detailed as follows
    @showprogress "ncomps" foreach(ncomps, iter_per_comp) do ncomp, max_iter
        @showprogress "iterations" for i in 1:max_iter
            tmp_targ = target .- rerotate(abs.(d), angles) 
            S = reconstruct(PCA(ncomp), tmp_targ; ref=ref)
            d = collapse!(target .- S, angles)
        end
    end
    return d
end

function ipca_rdi(target, angles, ref; ncomps=[1], iter_per_comp=[10])
    d = zeros(eltype(target), size(target, 1), size(target, 2))
    # for a given rank q, one iterative step is detailed as follows
    @showprogress "ncomps" foreach(ncomps, iter_per_comp) do ncomp, max_iter
        @showprogress "iterations" for i in 1:max_iter
            tmp_targ = target .- rerotate(abs.(d), angles) 
            refset = [tmp_targ ;;; ref]
            S = reconstruct(PCA(ncomp), tmp_targ; ref=refset)
            d = collapse!(target .- S, angles)
        end
    end
    return d
end

crop_size = 300
res_rdi = similar(cube, crop_size, crop_size, size(cube, 3))
for wl_idx in axes(res_rdi, 3)
    @info "Starting reduction for wavelength $wl_idx"
    target = crop(cube[:, :, wl_idx, :], crop_size)
    ref = crop(ref_cube[:, :, wl_idx, :], crop_size)
    ncomps = 10:20
    res_rdi[:, :, wl_idx] = ipca(target, angles, ref; ncomps, iter_per_comp=10:20)
end
DS9.set(res_rdi)
DS9.set(mask_circle(collapse(res_rdi), 12; fill=NaN))

save(datadir("20240729_HD169142_vampires_ipca_rdi.fits"), AstroImage(res_rdi))
save(datadir("20240729_HD169142_vampires_ipca_rdi_coll.fits"), AstroImage(collapse(res_rdi)))

res_ardi = similar(cube, crop_size, crop_size, size(cube, 3))
for wl_idx in axes(res_ardi, 3)
    @info "Starting reduction for wavelength $wl_idx"
    target = crop(cube[:, :, wl_idx, :], crop_size)
    ref = crop(ref_cube[:, :, wl_idx, :], crop_size)
    ncomps = 10:20
    res_ardi[:, :, wl_idx] = ipca_rdi(target, angles, ref; ncomps, iter_per_comp=10:20)
end
DS9.set(res_ardi)
DS9.set(mask_circle(collapse(res_ardi), 12; fill=NaN))

save(datadir("20240729_HD169142_vampires_ipca_adi+rdi.fits"), AstroImage(res_ardi))
save(datadir("20240729_HD169142_vampires_ipca_adi+rdi_coll.fits"), AstroImage(collapse(res_ardi)))




opt_nmf = PCA(10)
residual = similar(cube, (size(cube, 1), size(cube, 2), size(cube, 3)))
for wlidx in axes(cube, 3)
    cube_view = AnnulusView(cube[:, :, wlidx, :]; inner=8, outer=170)
    residual[:, :, wlidx] = process(Framewise(opt_nmf, delta_rot=1), cube_view, angles, fwhm=4, r=30)
end

save(datadir("20240729_HD169142_vampires_nmf$(opt_nmf.ncomps).fits"), AstroImage(residual))
# adi_nmf = collapse!(crop_cube .- reconstruct(opt_nmf, disk_less; angles=abaur_angles), abaur_angles)


# ncomps = 21:40
# resid_cubes2 = @showprogress map(n -> subtract(GreeDS(n), crop_cube; angles=abaur_angles), ncomps)
# resid_frames2 = map(cube -> collapse(cube, abaur_angles), resid_cubes2)
# resid_frames_cube2 = reduce((a,b) -> [a ;;; b], resid_frames2)

# fits.writeto(paths.data / "vampires_products" / "20230101_ABAur_median_adi.fits", median_adi_result, header=output_hdr, overwrite=True)

# resids_nmf = @showprogress map(n -> NMF(n)(crop_cube, abaur_angles), 1:40)
# resids_nmf_cube = reduce((a,b) -> [a ;;; b], resids_nmf)