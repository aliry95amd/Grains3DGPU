#!/usr/bin/env python3
import os
import sys
import math
import argparse
import scipy.special as sc
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


def analyticalSolution(v0, w0, gamma, tList):
    x = np.zeros((len(tList), 1))
    x = v0 / np.sqrt(w0 ** 2 - gamma ** 2) * np.exp(-gamma * tList) * np.sin(np.sqrt(w0 ** 2 - gamma ** 2) * tList)
    return x


def main():
    r = 0.1
    rho = 1000
    k = 50000
    en = 0.8
    parser = argparse.ArgumentParser(description='Generate plots for collidingSpheres from result files')
    parser.add_argument('--result-dir', dest='results_root', type=str, required=True,
                        help='Path to the per-test results directory (must contain collidingSpheres_position_y.dat)')
    parser.add_argument('--plot-dir', dest='plots_root', type=str, required=True,
                        help='Path where plots will be written')
    args = parser.parse_args()
    results_root = args.results_root
    plots_root = args.plots_root
    expected_name = 'collidingSpheres_position_y.dat'
    data_path = os.path.join(results_root, expected_name)
    if not os.path.exists(data_path):
        msg = f"Missing results file '{expected_name}' in directory '{results_root}'"
        print(msg, file=sys.stderr)
        raise FileNotFoundError(msg)
    data = np.loadtxt(data_path, delimiter=' ')
    numSolution = 2 * r - (data[:, 2] - data[:, 1])
    M = rho * 4.0 / 3.0 * math.pi * r * r * r
    w0 = np.sqrt(2.0 * k / M)
    gamma = -w0 * np.log(en) / np.sqrt(math.pi ** 2 + np.log(en) ** 2)
    contactTime = math.pi / np.sqrt(w0 ** 2 - gamma ** 2)
    analSolution = analyticalSolution(0.2, w0, gamma, data[:, 0])

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(data[::8, 0], numSolution[::8],
             linewidth=0, c="tab:blue", marker='o', markerfacecolor="tab:blue",
             label="Numerical solution")
    ax1.plot(data[:, 0], analSolution,
             linewidth=1, c="k",
             label="Analytical solution")
    ax1.set_xlabel(r"Time [s]")
    ax1.set_ylabel(r"Overlap [m]")
    ax1.set_xlim(0, contactTime)
    # ax1.set_ylim( 0, 0.04 )
    ax1.set_box_aspect(1)
    ax1.legend()
    ax1.grid(color='lightgrey', linestyle='--', linewidth=0.5)

    out_plot = os.path.join(plots_root, 'overlap.eps')
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.savefig(out_plot, format='eps', bbox_inches='tight')


if __name__ == '__main__':
    main()
