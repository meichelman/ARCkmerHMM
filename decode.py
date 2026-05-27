import argparse
from hmm_functions import read_HMM_parameters_from_file, emission_probabilities, calculate_posterior_probabilities, pmap_path, viterbi_path, write_posterior_probs, write_tracts
from helper_functions import load_obs_and_obs_rates, decoding_statistics
import numpy as np



def decode(obs_file, obs_rates_file, mode, param_file, decode_method, out_path_file, out_tracts_file):
    
    print('Loading data...')
    hmm_parameters = read_HMM_parameters_from_file(param_file)
    obs, obs_rates, contig_offsets, window_size, contig_lengths = load_obs_and_obs_rates(obs_file, obs_rates_file)

    print('-' * 40)
    print(f'> Number of windows: {len(obs)}')
    print(f'> Number of k-mers: {obs.astype(np.int64).sum()}')
    print('-' * 40)
    
    # Build contig slices from offsets
    contig_list = list(contig_offsets.keys())
    contig_slices = {}
    for i, contig in enumerate(contig_list):
        start = contig_offsets[contig]
        end = contig_offsets[contig_list[i + 1]] if i + 1 < len(contig_list) else len(obs)
        contig_slices[contig] = slice(start, end)
        
    all_obs = []
    all_obs_rates = []
    all_posterior_probs = []
    all_paths = []
    
    # emissions_probs = emission_probabilities(obs, obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions)
    # with open('emissions_probs.txt', 'w') as f:
    #     # Header: one column per state
    #     header = '\t'.join([f'state_{s}' for s in range(emissions_probs.shape[1])])
    #     f.write('window\t' + header + '\n')
        
    #     # One row per window
    #     for i in range(emissions_probs.shape[0]):
    #         row = '\t'.join(map(str, emissions_probs[i]))
    #         f.write(f'{i}\t' + row + '\n')

    
    print('Calculating posterior probabilities...')
    for contig, sl in contig_slices.items():
        contig_obs = obs[sl]
        contig_obs_rates = obs_rates[sl]

        emissions_probs = emission_probabilities(contig_obs, contig_obs_rates, hmm_parameters.emissions, hmm_parameters.dispersions, mode)
        posterior_probs = calculate_posterior_probabilities(emissions_probs, hmm_parameters)

        if decode_method == 'Viterbi':
            path = viterbi_path(emissions_probs, hmm_parameters)
        else:
            path = pmap_path(posterior_probs)

        all_obs.append(contig_obs)
        all_obs_rates.append(contig_obs_rates)
        all_posterior_probs.append(posterior_probs)
        all_paths.append(path)

    print('Writing output...')
    write_posterior_probs(
        contig_slices,
        np.concatenate(all_obs),
        np.concatenate(all_obs_rates),
        np.concatenate(all_posterior_probs, axis=-1),
        np.concatenate(all_paths),
        hmm_parameters,
        out_path_file,
        window_size,
        contig_lengths
    )
    write_tracts(
        contig_slices,
        np.concatenate(all_paths),
        hmm_parameters,
        out_tracts_file,
        window_size,
        contig_lengths
    )
    
    summary_stats = decoding_statistics(obs_file, out_tracts_file)
    
    print('-' * 40)
    print(f'> Input file length: {summary_stats[0]} bp')
    print(f'> Sum of archaic introgression calls: {summary_stats[1]} bp')
    print(f'> Archaic ancestry proportion: {summary_stats[2] * 100:.2f}%')
    print(f'> Number of tracts: {summary_stats[3]}')
    print(f'> Average tract length: {summary_stats[4]} bp')
    print(f'> Median tract length: {int(summary_stats[5])} bp')
    print(f'> Standard deviation of tract length: {int(summary_stats[6])} bp')
    print(f'> Mode tract length: {summary_stats[7]} bp')
    print(f'> Max tract length: {summary_stats[8]} bp')
    print(f'> Min tract length: {summary_stats[9]} bp')
    print('-' * 40)

    print('Done')
    
    return


def print_script_usage():
    toprint = f'''
    Hidden Markov Model for archaic human introgression inference using the ARCkmerFinder output.

    Usage:
    python decode.py -obs [obs_file] -obs_rates [obs_rates_file] -mode [nb|poisson] -param [hmm_parameters_file]
    python decode.py -obs [obs_file] -obs_rates [obs_rates_file] -mode [nb|poisson] -param [hmm_parameters_file] -viterbi -out_path [out_path_file] -out_tracts [out_tracts_file] 
        
    > HMM decoding                
        -obs                Input file with observation data (required)
        -obs_rates          Input file with observation rates estimates (required)
        -mode               Model type (default: nb (negative binomial), or poisson)
        -param              HMM parameters file (required)
        -out_path           Output file with decoded path (default: 'path.txt')
        -out_tracts         Output file with decoded tracts (default: 'tracts.txt')
    '''

    return toprint
    

def main():
    parser = argparse.ArgumentParser(description=print_script_usage(), formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("-obs",help="Input file with observation data (required)", type=str)
    parser.add_argument("-obs_rates", metavar='',help="Input file with observation rates estimates (required)", type=str)
    parser.add_argument("-mode", metavar='',help="Model type (default: nb (negative binomial), or poisson)", default='nb', type=str)
    parser.add_argument("-param", metavar='',help="HMM parameters file (required)", type=str)
    parser.add_argument("-viterbi", action='store_true', help="Decode using Viterbi algorithm (default: False)")
    parser.add_argument("-out_path", metavar='',help="Output file with decoded path (default: 'path.txt')", default = 'path.txt')
    parser.add_argument("-out_tracts", metavar='',help="Output file with decoded tracts (default: 'tracts.txt')", default = 'tracts.txt')

    args = parser.parse_args()
    
    if (args.obs is None) or (args.obs_rates is None) or (args.param is None) or (args.mode not in ['nb', 'poisson']):
        print(print_script_usage())
        return
    
    print('-' * 40)
    print(f'> Observations file: {args.obs}')
    print(f'> Observation rates file: {args.obs_rates}')
    print(f'> Model type: {args.mode}')
    print(f'> HMM parameters file: {args.param}')
    if args.viterbi:
        decode_method = 'Viterbi'
        print('> Decoding method: Viterbi algorithm')
    else:
        decode_method = 'PMAP'
        print('> Decoding method: PMAP')
    print(f'> Output file with decoded path: {args.out_path}')
    print(f'> Output file with decoded tracts: {args.out_tracts}')
    print('-' * 40)

    decode(args.obs, args.obs_rates, args.mode, args.param, decode_method, args.out_path, args.out_tracts)
    
    return
        

if __name__ == "__main__":
    main()
