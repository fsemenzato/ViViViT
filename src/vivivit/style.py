import shutil
from pathlib import Path

import matplotlib.pyplot as plt

REDSHIFTS = Path(__file__).resolve().parents[2]/'configs'/'lightcone_redshifts.txt'


def paper_style():
    plt.rcParams.update({'font.family': 'serif', 'legend.fontsize': 15, 'axes.labelsize': 20, 'axes.titlesize': 20, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'text.usetex': shutil.which('latex') is not None})
