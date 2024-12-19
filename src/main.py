from phaselib import fft_batch, OPD
from unwraplib import quality_guided
from auxlib import tiff_to_array_batch, get_files_recursive, plot_batch
import cv2


def main():
    BATCH_SIZE = 4
    MASK_RADIUS = 32
    WAVELENGTH = 700e-9
    DIR_PATH = r"off-axis-data/"

    all_files = get_files_recursive(DIR_PATH)
    for idx in range(0, len(all_files), BATCH_SIZE):
        ## INPUT HANDLEDED IN BATCHES
        file_path = all_files[idx : idx + BATCH_SIZE]
        array_batch = tiff_to_array_batch(file_path)
        plot_batch(array_batch, "this is array_batch")

        ## PHASE ALGORITHM
        phase_batch = fft_batch(array_batch, MASK_RADIUS)
        plot_batch(phase_batch, "this is phase_batch")

        ## UNWRAP
        unwrapped_phase_batch = quality_guided(phase_batch=phase_batch)
        plot_batch(unwrapped_phase_batch, "this is unwrapped_phase_batch")

        ##ODP
        height_map_batch = OPD(unwrapped_phase_batch, WAVELENGTH)
        plot_batch(height_map_batch, "this is height_map_batch")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
