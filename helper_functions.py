import numpy as np
from collections import defaultdict
import math



def load_obs_and_obs_rates(obs_file, obs_rates_file):
    
    # First pass: identify window_size from first non-zero count line
    window_size = 1000  # default window size if all counts are zero
    with open(obs_file) as infile:
        for line in infile:
            fields = line.split('\t')
            start, end, count = int(fields[1]), int(fields[2]), int(fields[3])
            if count > 0:
                window_size = end - start
                break
    
    # Build contig -> global obs index offset map AND per-contig window count
    contig_offsets = {}
    contig_lengths = {}
    contig_window_counts = defaultdict(int)
    offset = 0
    with open(obs_file) as infile:
        for line in infile:
            fields = line.split('\t')
            contig = fields[0]
            start, end, count = int(fields[1]), int(fields[2]), int(fields[3])
            
            if contig not in contig_offsets:
                contig_offsets[contig] = offset
                
            contig_lengths[contig] = max(contig_lengths.get(contig, 0), end)

            if count == 0:
                # Span covers multiple zero windows
                num_windows_in_run = math.ceil((end - start) / window_size)
                contig_window_counts[contig] += num_windows_in_run
                offset += num_windows_in_run
            else:
                contig_window_counts[contig] += 1
                offset += 1

    num_windows = offset
    obs_arr = np.zeros(num_windows, dtype=np.int16)
    obs_rates_arr = np.zeros(num_windows, dtype=float)
    # obs_rates_arr = np.zeros(num_windows, dtype=np.int16)

    # Load obs counts into array, using contig_offsets to determine global index
    global_idx = 0
    with open(obs_file) as infile:
        for line in infile:
            fields = line.split('\t')
            start, end, count = int(fields[1]), int(fields[2]), int(fields[3])

            if count == 0:
                num_windows_in_run = math.ceil((end - start) / window_size)
                global_idx += num_windows_in_run
            else:
                obs_arr[global_idx] = count
                global_idx += 1

    with open(obs_rates_file) as infile:
        for line in infile:
            if line.startswith('contig'):
                continue
            fields = line.strip().split('\t')
            contig, start, end, obs_rate = fields[0], int(fields[1]), int(fields[2]), float(fields[3])

            contig_offset = contig_offsets[contig]
            contig_end = contig_offset + contig_window_counts[contig]

            global_idx_start = contig_offset + start // window_size
            global_idx_end   = min(contig_offset + math.ceil(end / window_size), contig_end)

            obs_rates_arr[global_idx_start:global_idx_end] = obs_rate

    return obs_arr, obs_rates_arr, contig_offsets, window_size, contig_lengths


def get_observation_file_length(obs_file):
    assembly_length = 0
    with open(obs_file) as f:
        first_line = f.readline()
        prev_contig, prev_end = first_line.split('\t')[0], int(first_line.split('\t')[2])
        for line in f:
            cur_contig = line.split('\t')[0]
            if cur_contig != prev_contig:
                prev_contig = cur_contig
                assembly_length += prev_end
            prev_end = int(line.split('\t')[2])
    assembly_length += prev_end
    return assembly_length


def get_archaic_introgression_tracts(out_tracts_file):
    tracts = []
    with open(out_tracts_file) as f:
        f.readline() # skip header
        for line in f:
            fields = line.strip().split('\t')
            length, state = int(fields[3]), fields[4]
            if state == 'Archaic':
                tracts.append(length)
    return tracts
                


def decoding_statistics(obs_file, out_tracts_file):
    assembly_length = get_observation_file_length(obs_file)
    tracts = get_archaic_introgression_tracts(out_tracts_file)
    archaic_introgression_length = sum(tracts)
    proportion_introgressed = archaic_introgression_length / assembly_length if assembly_length > 0 else 0
    num_tracts = len(tracts)
    avg_tract_length = archaic_introgression_length / num_tracts if num_tracts > 0 else 0
    median_tract_length = np.median(tracts) if num_tracts > 0 else 0
    std_tract_length = np.std(tracts) if num_tracts > 0 else 0
    mode_tract_length = max(set(tracts), key=tracts.count) if num_tracts > 0 else 0
    max_tract_length = max(tracts) if num_tracts > 0 else 0
    min_tract_length = min(tracts) if num_tracts > 0 else 0
    return (assembly_length, archaic_introgression_length, proportion_introgressed,
            num_tracts, avg_tract_length, median_tract_length,
            std_tract_length, mode_tract_length, max_tract_length,
            min_tract_length)