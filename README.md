# STEPS OF THE CODE

## take image input

The image is taken as 2darray.

## preprocess images

The image is normalized and then noise reduction is applied.

## phase extraction (with masking)

Fft is applied then masking is applied discarding the DC and the conjugate term

## unwrap phase

Phase is unwrapped

## phase to dh

OPD is calculated for dh

## plot

The outputs are plotted

## performance increase options

Can be parallelized with batches and shit. Saving the input as 3darray as a batch of 2d images.
ROI can be used before everything else to save compute.
Filtering can be after fft for frequency domain filtering.
GPU acceleration. (cuPy)
