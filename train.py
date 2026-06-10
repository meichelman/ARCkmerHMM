import argparse
from hmm_functions import read_HMM_parameters_from_file, train_model, write_HMM_to_file
from helper_functions import load_obs_and_obs_rates
import numpy as np



def train(obs_file, obs_rates_file, mode, param_file, out_file):
    
    print('Loading data...')
    hmm_parameters = read_HMM_parameters_from_file(param_file)
    obs, obs_rates, _, _, _ = load_obs_and_obs_rates(obs_file, obs_rates_file)
    # print(np.where(obs_rates == 0))

    print('-' * 40)
    print(f'> Number of windows: {len(obs)}')
    print(f'> Number of k-mers: {obs.astype(np.int64).sum()}')
    print('-' * 40)

    if obs.astype(np.int64).sum() == 0:
        print('All observations are zero. Skipping training.')
    elif obs.astype(np.int64).sum() > 0:
        print('Training HMM...')
        hmm_parameters = train_model(obs, obs_rates, hmm_parameters, mode)
    print('Writing output...')
    write_HMM_to_file(hmm_parameters, out_file)
    print('Done')
    
    return


def print_script_usage():
    toprint = f'''
    Hidden Markov Model for archaic human introgression inference using the ARCkmerFinder output.

    Usage:
    python train.py -obs [obs_file] -obs_rates [obs_rates_file] -mode [nb|poisson] 
    python train.py -obs [obs_file] -obs_rates [obs_rates_file] -mode [nb|poisson] -param [hmm_parameters_file] -out [output_file]
    
    > Train the HMM
        -obs                Input file with observation data (required)
        -obs_rates          Input file with observation rates estimates (required)
        -mode               Model type (default: nb (negative binomial), or poisson)
        -param              HMM parameters file (default: human/neanderthal-like parameters)
        -out                Output file with trained HMM parameters (default: 'trained.json')
        
    '''

    return toprint


def main():
    parser = argparse.ArgumentParser(description=print_script_usage(), formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("-obs",help="Input file with observation data (required)", type=str)
    parser.add_argument("-obs_rates", metavar='',help="Input file with observation rates estimates (required)", type=str)
    parser.add_argument("-mode", metavar='',help="Model type (default: nb (negative binomial), or poisson)", default='nb', type=str)
    parser.add_argument("-param", metavar='',help="HMM parameters file (default: human/neanderthal-like parameters)", type=str)
    parser.add_argument("-out", metavar='',help="Output file with trained HMM parameters (default: 'trained.json')", default = 'trained.json')
    
    args = parser.parse_args()
    
    if (args.obs is None) or (args.obs_rates is None) or (args.mode not in ['nb', 'poisson']):
        print(print_script_usage())
        return
    
    print('-' * 40)
    print(f'> Observations file: {args.obs}')
    print(f'> Observation rates file: {args.obs_rates}')
    print(f'> Model type: {args.mode}')
    print(f'> HMM parameters file: {args.param if args.param else "using default human/neanderthal-like parameters"}')
    print(f'> Output file for trained parameters: {args.out}')
    print('-' * 40)

    train(args.obs, args.obs_rates, args.mode,args.param, args.out)
    
    return
        

if __name__ == "__main__":
    main()
