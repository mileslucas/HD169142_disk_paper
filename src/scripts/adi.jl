using AstroImages
using ADI
using HCIToolbox
using ProgressMeter
using Statistics
using BiweightStats

datadir(args...) = abspath(joinpath(@__DIR__, "..", "data", args...))

cube = AstroImage(datadir("20240729_HD169142_vampires_adi_cube.fits"))
angles = AstroImage(datadir("20240729_HD169142_vampires_adi_angles.fits"))


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