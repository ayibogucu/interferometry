import lib.phase as phase
import lib.unwrap as unwrap
import lib.aux as aux
import lib.opd as opd


BATCH_SIZE = 4
MASK_RADIUS = 32
WAVELENGTH = 700e-9
DIR_PATH = r"off-axis-data/"


def main():
    all_files = aux.get_files_recursive(DIR_PATH)
    for idx in range(0, len(all_files), BATCH_SIZE):
        ## INPUT HANDLEDED IN BATCHES
        file_path = all_files[idx : idx + BATCH_SIZE]
        array_batch = aux.tiff_to_array_batch(file_path)
        aux.plot_batch(array_batch)

        ## PHASE ALGORITHM
        phase_batch = phase.fft_batch(array_batch, MASK_RADIUS)
        aux.plot_batch(phase_batch)

        ## UNWRAP
        unwrapped_phase_batch = unwrap.quality_guided(phase_batch=phase_batch)
        aux.plot_batch(unwrapped_phase_batch)

        ##ODP
        height_map_batch = opd.OPD(unwrapped_phase_batch, WAVELENGTH)
        aux.plot_batch_3d_heightmaps(height_map_batch)


if __name__ == "__main__":
    main()
