# TODO: it is poor practise to maintain multiple grid generating scripts.
import argparse
import os
import subprocess
import pathlib

multiclass = False
opts_dict = {}
total_jobs = 1


def add_opt(key, val):
    global total_jobs
    total_jobs *= len(val)
    opts_dict[key] = val


# ## 2D to 2D experiments
# add_opt('base_dist',
#         ['sine_wave', 'four_circles', 'diamond', 'checkerboard'])
# add_opt('optim',
#         ['Adam'])
# add_opt('model', ['dense'])
# add_opt('inner_activ',
#         ['selu'])
# add_opt('load', [0])
# add_opt('gclip', [0])
# add_opt('scheduler', ['cosine'])
#
# # add_opt('latent_dim', list(range(2, 22, 2)) * 10)
# # add_opt('latent_dim', [20])
# add_opt('latent_dim', [2])
# add_opt('target_dim', [2])
# add_opt('chain_dims', [0])
# add_opt('test_norm', [0])
# add_opt('get_kl', [0])
# add_opt('get_sinkhorn', [0])
# # import numpy as np
# # # add_opt('std', list(np.exp(np.linspace(np.log(0.01), np.log(0.15), 20))) * 3)
# # add_opt('std', list(np.exp(np.linspace(np.log(0.01), np.log(0.15), 20))))
#
# add_opt('noise_strength', [0])
# add_opt('epochs', [1000])
# add_opt('lr', [0.001])
# add_opt('target', ['normal'])
# add_opt('activation', ['none'])
# add_opt('batch_size', [1000])
# add_opt('stochastic', [0])
# add_opt('depth', ['paper2'])
# add_opt('nsteps_train', [100])  # For some datasets this must be a multiple of 8 or 4
# add_opt('nrun', [1])
# add_opt('n_test', [20])
# add_opt('ncalc', [100000])
# add_opt('dist_measure', ['sinkhorn_fast'])
# add_opt('final_plot_encoding', [1])
# add_opt('det_only', [0])
# add_opt('auto_J', [0])
# add_opt('beta_j', [0])
# add_opt('true_likelihood', [0])


# # Dimensionality reduction
# # add_opt('base_dist',
# #         ['checkerboard'])
# # add_opt('base_dist',
# #         ['shells'])
# add_opt('base_dist',
#         ['stars'])
# add_opt('std', [0.05])
# add_opt('latent_dim', list(range(2, 22, 2)) * 10)
# # add_opt('latent_dim', list(range(14, 22, 2)) * 20)
# # add_opt('latent_dim', [2, 4, 8, 20])

# add_opt('base_dist',
#         ['stars'])
# add_opt('base_dist',
#         ['shells'])
# add_opt('latent_dim', [20])
# import numpy as np
# # add_opt('std', list(np.exp(np.linspace(np.log(0.01), np.log(0.15), 20))) * 5)
# add_opt('std', list(np.exp(np.linspace(np.log(0.01), np.log(0.15), 10))) * 10)
# # add_opt('std', [0.01])
# # add_opt('std', [0.01, 0.02, 0.05, 0.1])
#
# add_opt('final_plot_encoding', [0])
# # add_opt('final_plot_encoding', [1])
#
# add_opt('optim',
#         ['Adam'])
# add_opt('model', ['dense'])
# add_opt('inner_activ',
#         ['selu'])
# add_opt('gclip', [0])
# add_opt('epochs', [200])
# add_opt('train_ae', [100])
# # add_opt('epochs', [1200])
# # add_opt('train_ae', [800])
# add_opt('beta_dist', [1])
# add_opt('lr', [0.001])
# add_opt('inorm', [0])
#
# # Defaults
# add_opt('get_kl', [1])
# add_opt('multi_jobs', [1])
# add_opt('load', [0])
# add_opt('det_only', [0])
# add_opt('auto_J', [0])
# add_opt('beta_j', [0])
# add_opt('true_likelihood', [0])
# add_opt('target_dim', [2])
# add_opt('chain_dims', [0])
# add_opt('test_norm', [0])
# add_opt('get_sinkhorn', [0])
# add_opt('noise_strength', [0])
# add_opt('target', ['normal'])
# add_opt('activation', ['none'])
# add_opt('batch_size', [1000])
# add_opt('stochastic', [0])
# add_opt('depth', ['paper2'])
# add_opt('nsteps_train', [100])  # For some datasets this must be a multiple of 8 or 4
# add_opt('dist_measure', ['sinkhorn_fast'])


# # Dimensionality preservation experiment
# # add_opt('test_norm', [1])
# add_opt('chain_dims', [1])
# add_opt('get_kl', [0])
# add_opt('target_dim', [10, 20])
# # add_opt('target_dim', list(range(2, 22, 2)) * 10)
# add_opt('get_sinkhorn', [1])

# # ND sinkhorn training
# # add_opt('latent_dim', list(range(2, 22, 2)))
# add_opt('latent_dim', [2, 20])
# add_opt('batch_size', [1000])
# add_opt('sigma', [0.9, 0.95, 0.99])
# add_opt('nrun', [10000])


# AAE
add_opt('inner_activ',
        ['selu'])
add_opt('optim',
        ['Adam'])
# add_opt('model', ['dense'])
# add_opt('model', ['splinet'])
add_opt('model', ['nsf'])
add_opt('load', [0])

add_opt('noise_strength', [0])
add_opt('epochs', [100])
add_opt('lr', [0.0001])
add_opt('base_dist', ['checkerboard'])
add_opt('target', ['normal'])
add_opt('batch_size', [1000])
add_opt('activation', ['none'])
add_opt('depth', ['paper2'])
# add_opt('nsteps_train', [100])
# add_opt('t_disc', [2])
add_opt('t_disc', [2])
add_opt('beta_j', [0, 0.1, 0.5, 1])


#
# # Dimensionality preservation experiment
# # add_opt('test_norm', [1])
# add_opt('chain_dims', [1])
# # add_opt('target_dim', list(range(2, 22, 2)) * 10)
# add_opt('target_dim', [20])

# ## Reset optimizer experiment
# add_opt('base_dist', ['checkerboard'])
# add_opt('dist_measure', ['sinkhorn'])
# add_opt('batch_size', [1000])
# add_opt('epochs', [150])
# add_opt('reset_optimizer', [100])
# add_opt('optim', ['Adam'])
# add_opt('lr', [0.001])
# add_opt('wd', [0.001])
# add_opt('p', [0.001])
# add_opt('reset_lr', [0.1, 0.01, 0.001, 0.0001])
# add_opt('reset_wd', [0.001])
# add_opt('reset_p', [0.001])
# add_opt('scheduler', ['cosine', 'none', 'plateau'])
#
#
# # Defaults for this experiment
# add_opt('inner_activ', ['selu'])
# add_opt('target', ['normal'])
# add_opt('model', ['dense'])
# add_opt('load', [0])
# add_opt('latent_dim', [2])
# add_opt('target_dim', [2])
# add_opt('noise_strength', [0])
# add_opt('train_ae', [0])
# add_opt('activation', ['none'])
# add_opt('bnorm', [0])
# add_opt('lnorm', [0])
# add_opt('depth', ['paper2'])
# add_opt('stochastic', [0])

# ## Retrain spline experiment
# add_opt('base_dist', ['checkerboard'])
# add_opt('dist_measure', ['sinkhorn'])
# add_opt('batch_size', [1000])
# add_opt('epochs', [100])
# add_opt('reset_optimizer', [0])
# add_opt('optim', ['Adam'])
# add_opt('lr', [0.001, 0.0001, 0.00001, 0.000001])
# add_opt('scheduler', ['none'])
# add_opt('weight_noise', [0.005, 0.01, 0.03, 0.05])
#
#
# # Defaults for this experiment
# add_opt('inner_activ', ['selu'])
# add_opt('target', ['normal'])
# add_opt('model', ['trained_spline'])
# add_opt('load', [0])
# add_opt('latent_dim', [2])
# add_opt('target_dim', [2])
# add_opt('noise_strength', [0])
# add_opt('train_ae', [0])
# add_opt('activation', ['none'])
# add_opt('bnorm', [0])
# add_opt('lnorm', [0])
# add_opt('depth', ['paper2'])
# add_opt('stochastic', [0])


# ## Get derivatives and plot for the INN experiments
# add_opt('base_dist', ['checkerboard'])
# add_opt('dist_measure', ['sinkhorn'])
# add_opt('batch_size', [10, 100, 500, 6000, 10000])
# add_opt('get_mle', [0])
# add_opt('n_sample', [int(1e6)])
#
#
# # Defaults for this experiment
# add_opt('target', ['normal'])
# add_opt('model', ['nsf'])
# add_opt('latent_dim', [2])
# add_opt('output_dim', [2])

# ## Train with the jacobian loss term
# # add_opt('target', ['sine_wave', 'four_circles', 'diamond', 'checkerboard'])
# # add_opt('base_dist', ['normal'])
# add_opt('base_dist', ['sine_wave', 'four_circles', 'diamond', 'checkerboard'])
# # add_opt('base_dist', ['checkerboard'])
# add_opt('target', ['normal'])
# add_opt('dist_measure', ['sinkhorn_fast'])
# add_opt('batch_size', [100])
# add_opt('epochs', [1000])
# add_opt('lr', [0.00001])
# add_opt('scheduler', ['cosine'])
# # add_opt('beta_j', [1, 0.1, 0.01, 0.001, 0.0001])
# add_opt('beta_j', [1])
# add_opt('load', [1])
# add_opt('retrain', [0])
# add_opt('model', ['ni_nsf'])
# # add_opt('model', ['dense'])
# add_opt('det_only', [1])
# add_opt('auto_J', [1])
# add_opt('true_likelihood', [0])
# add_opt('bias', [1])
# add_opt('activation', ['tanh'])
# add_opt('gclip', [2])
# add_opt('depth', ['paper2'])
# # add_opt('depth', ['def'])
# # add_opt('sw', [64, 128, 256, 512, 1024])
# # add_opt('sd', [1])
# add_opt('final_plot_encoding', [1])
# add_opt('bnorm', [0])
# add_opt('lnorm', [0])
# add_opt('inorm', [0])
#
# # Defaults for this experiment
# add_opt('optim', ['Adam'])
# add_opt('reset_optimizer', [0])
# add_opt('latent_dim', [2])
# add_opt('target_dim', [2])
# add_opt('noise_strength', [0])
# add_opt('train_ae', [0])
# add_opt('inner_activ', ['selu'])
# add_opt('stochastic', [0])

# ## Retrain spline experiment
# add_opt('base_dist', ['checkerboard'])
# add_opt('dist_measure', ['sinkhorn'])
# add_opt('batch_size', [1000])
# add_opt('epochs', [100])
# add_opt('optim', ['Adam'])
# add_opt('lr', [0.0001])
# add_opt('scheduler', ['none'])
# add_opt('inner_activ', ['selu', 'elu'])
# add_opt('sw', [2])
# add_opt('sd', [50, 100, 500, 1000])
# add_opt('bias', [0, 1])
#
# # Defaults for this experiment
# add_opt('reset_optimizer', [0])
# add_opt('target', ['normal'])
# add_opt('model', ['dense'])
# add_opt('load', [0])
# add_opt('latent_dim', [2])
# add_opt('target_dim', [2])
# add_opt('noise_strength', [0])
# add_opt('train_ae', [0])
# add_opt('activation', ['none'])
# add_opt('bnorm', [0])
# add_opt('lnorm', [0])
# add_opt('depth', ['def'])
# add_opt('stochastic', [0])


# ## Train exclusively on distribution matching
# # add_opt('target', ['sine_wave', 'four_circles', 'diamond', 'checkerboard'])
# # add_opt('base_dist', ['normal'])
# # add_opt('base_dist', ['sine_wave', 'four_circles', 'diamond', 'checkerboard'])
# add_opt('base_dist', ['checkerboard_modes'])
# add_opt('target', ['normal'])
# add_opt('dist_measure', ['sinkhorn_fast', 'mmd'])
# add_opt('batch_size', [1000])
# add_opt('epochs', [100])
# add_opt('lr', [0.001])
# add_opt('scheduler', ['cosine'])
# add_opt('load', [0])
# add_opt('retrain', [0])
# add_opt('model', ['dense'])
# add_opt('activation', ['none'])
# add_opt('gclip', [0, 5])
#
# # Defaults for this experiment
# add_opt('true_likelihood', [0])
# add_opt('bias', [1])
# add_opt('det_only', [0])
# add_opt('auto_J', [0])
# add_opt('beta_j', [0])
# add_opt('optim', ['Adam'])
# add_opt('reset_optimizer', [0])
# add_opt('latent_dim', [2])
# add_opt('target_dim', [2])
# add_opt('noise_strength', [0])
# add_opt('train_ae', [0])
# add_opt('depth', ['paper2'])
# add_opt('inner_activ', ['selu'])
# add_opt('bnorm', [0])
# add_opt('lnorm', [0])
# add_opt('stochastic', [0])


# ## Lazy Friday AAE
# add_opt('beta_j', [0.01, 0.001, 0.0001])
# add_opt('epochs', [100])
# add_opt('batch_size', [1000])
# add_opt('model', ['nsf'])
# add_opt('base_dist', ['checkerboard'])
# add_opt('wa', [0])


# # # AAE Dimensionality reduction experiments
# add_opt('base_dist',
#         ['checkerboard'])
# # add_opt('base_dist',
# #         ['shells'])
# # add_opt('base_dist',
# #         ['stars'])
# add_opt('std', [0.05])
# # add_opt('latent_dim', list(range(2, 22, 2)) * 10)
# # add_opt('latent_dim', list(range(2, 22, 2)))
# add_opt('latent_dim', [10])
#
# # add_opt('base_dist',
# #         ['stars'])
# # # add_opt('base_dist',
# # #         ['shells'])
# # add_opt('latent_dim', [20])
# # import numpy as np
# # # add_opt('std', list(np.exp(np.linspace(np.log(0.01), np.log(0.15), 20))) * 5)
# # add_opt('std', list(np.exp(np.linspace(np.log(0.01), np.log(0.15), 10))))
#
# add_opt('optim',
#         ['Adam'])
# add_opt('model', ['dense'])
# add_opt('inner_activ',
#         ['selu'])
# add_opt('gclip', [0])
# add_opt('epochs', [100])
# add_opt('train_ae', [0])
# add_opt('beta_dist', [1])
# add_opt('lr', [0.001])
# add_opt('wa', [0])
#
# # Defaults
# add_opt('get_kl', [1])
# add_opt('load', [0])
# add_opt('beta_j', [0])
# add_opt('target_dim', [2])
# add_opt('chain_dims', [0])
# add_opt('noise_strength', [0])
# add_opt('target', ['normal'])
# add_opt('activation', ['none'])
# add_opt('batch_size', [1000])
# add_opt('depth', ['paper2'])
# add_opt('nsteps_train', [100])  # For some datasets this must be a multiple of 8 or 4


def _get_args():
    parser = argparse.ArgumentParser()
    ## General settings for slurm
    parser.add_argument('-d', '--outputdir', type=str,
                        help='Choose the base output directory', required=True)
    parser.add_argument('-n', '--outputname', type=str,
                        help='Set the output name directory')
    parser.add_argument('--squeue', type=str, default='shared-gpu,private-dpnc-gpu')
    # parser.add_argument('--squeue', type=str, default='shared-cpu,private-dpnc-cpu')
    # parser.add_argument('--stime', type=str, default='00-04:00:00')
    parser.add_argument('--stime', type=str, default='00-12:00:00')
    parser.add_argument('--smem', type=str, default='10GB')
    parser.add_argument('--work-dir', type=str, default='/home/users/k/kleins/MLproject/distribution_matching/')
    parser.add_argument('--submit', action='store_true',
                        dest='submit')
    parser.add_argument('--sbatch-output', type=str, default='submit.txt')
    parser.add_argument('--singularity-instance', type=str,
                        default='/home/users/k/kleins/MLproject/distribution_matching/container/pytorch.sif')
    parser.add_argument('--singularity-mounts', type=str,
                        default=None)
    # parser.add_argument('--singularity-mounts', type=str,
    #                     default='/home/users/k/kleins/scratch:/mnt/scratch,/home/users/k/kleins/atlas:/mnt/atlas, /home/users/k/kleins/scratch:/mnt/scratch')
    parser.add_argument('--experiment', type=str,
                        default='tests_no_recon.py')
    parser.set_defaults(submit=False)
    return parser.parse_args()


def main():
    args = _get_args()
    print('Generating gridsearch with {} subjobs'.format(total_jobs))
    runfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.experiment)
    cmd = '\nsrun singularity exec --nv'
    if args.singularity_mounts is not None:
        cmd += ' -B {}'.format(args.singularity_mounts)
    cmd += ' {0}\\\n\tpython3 {1} -d {2} -n {3}_${{SLURM_ARRAY_TASK_ID}} \\\n\t\t'.format(
        # --train-data {4}\\\n\t\t'.format(
        args.singularity_instance,
        runfile,
        args.outputdir,
        args.outputname)
    # args.train_data)
    # if args.val_data is not None:
    #     cmd += '--val-data {}\\\n\t\t'.format(args.val_data)
    pathlib.Path(args.sbatch_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.sbatch_output, 'w') as f:
        f.write('''#!/bin/sh
#SBATCH --job-name={0}
#SBATCH --cpus-per-task=1
#SBATCH --time={1}
#SBATCH --partition={2}
#SBATCH --output=/home/users/k/kleins/MLproject/distribution_matching/jobs/slurm-%A-%x_%a.out
#SBATCH --chdir={3}
#SBATCH --mem={4}
#SBATCH --gpus=1
#SBATCH -a 0-{5}
export XDG_RUNTIME_DIR=""
module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12\n'''.format(
            args.outputname,
            args.stime,
            args.squeue,
            args.work_dir,
            args.smem,
            total_jobs - 1
        ))
        running_total = 1
        for opt, vals in opts_dict.items():
            f.write('{}=({}'.format(opt.replace('-', ''), vals[0]))
            for val in vals[1:]:
                f.write(' {}'.format(val))
            f.write(')\n')
            cmd += '--{0} ${{{1}[`expr ${{SLURM_ARRAY_TASK_ID}} / {2} % {3}`]}}\\\n\t\t'.format(opt,
                                                                                                opt.replace('-', ''),
                                                                                                running_total,
                                                                                                len(vals))
            running_total *= len(vals)
        if multiclass:
            cmd += '--m\\\n\t\t'
        # if abseta:
        #     cmd += '-eta\\\n\t\t'
        cmd += '\n\n'
        f.write(cmd)
    if args.submit is True:
        subprocess.run(['sbatch', '{}'.format(args.sbatch_output)])
    return


#     TODO: You need to save the hyper parameters used to call each of the models somewhere!


if __name__ == '__main__':
    main()
