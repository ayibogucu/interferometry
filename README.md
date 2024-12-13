# STEPS OF THE CODE

## Take image input

The image batch is taken as 3darray.

## Preprocess images

The image is normalized and then noise reduction is applied.

## Phase Algorithms

## Unwrap phase

## Calculate ODP

$$ \alpha = 2\pi \cdot OPD \cdot  \frac{\Delta v}{c} $$

## Plot

The outputs are plotted

## Error sources

## Output increase options

## Performance increase options

Can be parallelized with batches and shit.
Saving the input as 3darray as a batch of 2d images.
ROI can be used before everything else to save compute.
Filtering can be after fft for frequency domain filtering.
GPU acceleration. (cuPy)
