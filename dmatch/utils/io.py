import pathlib


def on_cluster():
    """
    :return: True if running job on cluster
    """
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    if id == 'users':
        return True
    else:
        return False


def get_top_dir():
    p = pathlib.Path().absolute()
    id = p.parts[:3][-1]
    # At least one 'sv_ims' must be set to reproduce experiments.
    if id == 'samklein':
        sv_ims = '/Users/samklein/PycharmProjects/distribution_matching'
    elif id == 'users':
        sv_ims = '/home/users/k/kleins/MLproject/distribution_matching'
    else:
        raise ValueError('Unknown path for saving images {}'.format(p))
    return sv_ims
