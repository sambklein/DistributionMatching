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


# AE INN settings
# add_opt('latent_dim', list(range(6, 22, 2)))
add_opt('latent_dim', [2])
add_opt('output_dim', [2])
add_opt('lr', [0.0001])
add_opt('epochs', [1000])
add_opt('nrun', [100])
add_opt('ncalc', [int(1e6)])
add_opt('bnorm', [0])
add_opt('dropout', [0])
add_opt('depth', ['paper'])
add_opt('model', ['nsf', 'dense'])
# add_opt('base_dist', ['checkerboard', 'diamond', 'four_circles'])
add_opt('base_dist', ['checkerboard'])
add_opt('target', ['normal'])
add_opt('learn_spline', [1])
add_opt('beta_j', [0, 10, 0.1, 0.01, 0.001])


def _get_args():
    parser = argparse.ArgumentParser()
    ## General settings for slurm
    parser.add_argument('-d', '--outputdir', type=str,
                        help='Choose the base output directory',
                        required=True)
    parser.add_argument('-n', '--outputname', type=str,
                        help='Set the output name directory')
    # parser.add_argument('--save-arch', action='count',
    #                     help='Save the NN architectures to json file')
    # parser.add_argument('--train-data', type=str,
    #                     help='File containing training data',
    #                     required=True)
    # parser.add_argument('--val-data', type=str,
    #                     help='File containing validation data. If not provided, training data will be split.',
    #                     )
    parser.add_argument('--squeue', type=str, default='shared-gpu,private-dpnc-gpu')
    parser.add_argument('--stime', type=str, default='00-10:00:00')
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
                        default='fully_supervised.py')
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
