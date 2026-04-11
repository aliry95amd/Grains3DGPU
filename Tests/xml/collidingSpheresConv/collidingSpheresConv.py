#!/usr/bin/env python3
"""Post-processing script for the collidingSpheresConv convergence test.

Setup:
  Sphere 1 at Z=0.4 m, Vz=-1.0 m/s approaches Sphere 2 at Z=0.0 m, Vz=0.
  Gravity GZ=-9.81 m/s².  Elastic Hooke contact (kn=5e4 N/m, en=1).

Key insight (why gravity is needed to distinguish integrators):
  The relative coordinate x = z1-z2-2R satisfies ẍ = -ω²x regardless of
  gravity (gravity cancels between equal-mass spheres), so both integrators
  give essentially the same O(dt²) error during contact.  The difference
  arises in the free-flight phases: the CM falls under constant gravity g,
  and FirstOrderExplicit accumulates O(dt) error there, while
  SecondOrderLeapFrog (KDK Verlet) is exact for constant acceleration.

Analytical solution (all motion in Z, g = -9.81 m/s²):

  Phase 1 – free approach [0, t0=0.2 s]:
    z1(t) = Z1_INIT + V0*t   + 0.5*g*t²
    z2(t) = Z2_INIT + V2I*t  + 0.5*g*t²
    Contact at t0 = (Z1_INIT - Z2_INIT - 2R)/(V2I - V0) = 0.2 s (gravity
    cancels in relative coord, so t0 is unchanged from no-gravity case).

  Phase 2 – contact [t0, t0+Tc]:
    ω  = sqrt(2*kn/m),  Tc = π/ω   (relative dynamics unaffected by gravity)
    x_rel(τ) = (V_rel/ω)*sin(ω*τ)  [τ = t - t0, V_rel = v1(t0)-v2(t0)]
    z_cm(τ)  = z_cm0 + v_cm0*τ + 0.5*g*τ²
    z1 = z_cm + R + x_rel/2,   z2 = z_cm - R - x_rel/2

  Phase 3 – post-contact [t0+Tc, Tend=0.3 s]:
    (elastic equal-mass: velocities exchange in CM frame)
    z1(t) = z1_end + v1_post*s + 0.5*g*s²
    z2(t) = z2_end + v2_post*s + 0.5*g*s²
    where s = t - t0 - Tc.

Expected convergence:
  - FirstOrderExplicit  : O(dt)  — free-flight under gravity accumulates error
  - SecondOrderLeapFrog : O(dt²) — exact for constant acceleration
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": 16})

# ---------------------------------------------------------------------------
# Physical parameters (must match the YAML)
# ---------------------------------------------------------------------------
R      = 0.1       # sphere radius               [m]
RHO    = 1000.0    # sphere density              [kg/m³]
KN     = 5.0e4     # normal spring stiffness     [N/m]
V0     = -1.0      # initial velocity of sphere 1 [m/s]
V2I    = 0.0       # initial velocity of sphere 2 [m/s]
G      = -9.81     # gravitational acceleration  [m/s²]
T_END  = 0.3       # simulation end time          [s]

Z1_INIT = 0.4      # initial z of sphere 1
Z2_INIT = 0.0      # initial z of sphere 2

# Derived quantities
M     = RHO * (4.0 / 3.0) * np.pi * R ** 3   # ≈ 4.18879 kg
OMEGA = np.sqrt(2.0 * KN / M)                 # relative-motion angular freq ≈ 154.53 rad/s
T_C   = np.pi / OMEGA                         # contact duration ≈ 0.0203 s

# Contact time: gravity cancels in relative coordinate, so t0 is same as no-gravity
T0    = (Z1_INIT - Z2_INIT - 2.0 * R) / (V2I - V0)   # = 0.2 s

# State just before contact (velocities include gravity free-fall)
V1_AT_T0  = V0  + G * T0
V2_AT_T0  = V2I + G * T0
Z1_AT_T0  = Z1_INIT + V0  * T0 + 0.5 * G * T0 ** 2
Z2_AT_T0  = Z2_INIT + V2I * T0 + 0.5 * G * T0 ** 2

V_CM0 = (V1_AT_T0 + V2_AT_T0) / 2.0   # CM velocity at contact start
Z_CM0 = (Z1_AT_T0 + Z2_AT_T0) / 2.0   # CM position at contact start
V_REL = V1_AT_T0 - V2_AT_T0            # relative velocity at contact (= V0 - V2I = -1 m/s)

# End-of-contact state (tau = Tc)
Z_CM_END = Z_CM0 + V_CM0 * T_C + 0.5 * G * T_C ** 2
V_CM_END = V_CM0 + G * T_C
# x_rel(Tc) = 0 (half oscillation), dx_rel/dt(Tc) = -V_REL (elastic)
Z1_END   = Z_CM_END + R
Z2_END   = Z_CM_END - R
V1_POST  = V_CM_END + (-V_REL) / 2.0   # CM vel + half of exchanged relative vel
V2_POST  = V_CM_END - (-V_REL) / 2.0

# Test matrix
DT_VALUES  = [1e-3, 1e-4, 1e-5, 1e-6]
DT_LABELS  = ["dt1e-3", "dt1e-4", "dt1e-5", "dt1e-6"]
FINEST_DT_LABEL = "dt1e-6"

# Colours and markers.  Stagger trajectory markers so they do not overlap.
# Finest dt=1e-6 and T_END=0.3 → 300 000 points; stride 5000 gives ~60 markers.
INTEGRATORS = {
    #  key            (display label,                              colour,      marker, traj-slice)
    "FirstOrder":  (r"$1^{\mathrm{st}}$-order explicit",
                    "tab:blue",   "o", slice(0,    None, 5000)),
    "SecondOrder": (r"$2^{\mathrm{nd}}$-order leap-frog",
                    "tab:blue", "X", slice(2500, None, 5000)),
}


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------
def analytical_z(t: np.ndarray):
    """Return (z1_anal, z2_anal) for array of absolute times (with gravity)."""
    z1 = np.empty_like(t)
    z2 = np.empty_like(t)

    for i, ti in enumerate(t):
        if ti <= T0:
            # Phase 1: free fall under gravity, no contact force
            z1[i] = Z1_INIT + V0  * ti + 0.5 * G * ti ** 2
            z2[i] = Z2_INIT + V2I * ti + 0.5 * G * ti ** 2
        elif ti <= T0 + T_C:
            # Phase 2: harmonic contact (relative) + CM parabola (gravity)
            tau   = ti - T0
            x_rel = (V_REL / OMEGA) * np.sin(OMEGA * tau)   # z1 - z2 - 2R
            z_cm  = Z_CM0 + V_CM0 * tau + 0.5 * G * tau ** 2
            z1[i] = z_cm + R + x_rel / 2.0
            z2[i] = z_cm - R - x_rel / 2.0
        else:
            # Phase 3: both spheres in free fall again after elastic exchange
            s     = ti - T0 - T_C
            z1[i] = Z1_END + V1_POST * s + 0.5 * G * s ** 2
            z2[i] = Z2_END + V2_POST * s + 0.5 * G * s ** 2

    return z1, z2


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------
def load_data(results_root: str, root_name: str) -> np.ndarray:
    """Load *_position_z.dat; returns array with columns [time, z_sphere1, z_sphere2]."""
    fname = os.path.join(results_root, f"{root_name}_position_z.dat")
    if not os.path.exists(fname):
        msg = (f"Missing results file '{os.path.basename(fname)}' "
               f"in '{results_root}'")
        print(msg, file=sys.stderr)
        raise FileNotFoundError(msg)
    return np.loadtxt(fname)


# ---------------------------------------------------------------------------
# Trajectory plot  (finest dt, both integrators)
# ---------------------------------------------------------------------------
def plot_trajectory(results_root: str, plots_root: str) -> None:
    """Z(t) for both spheres (finest dt, both integrators) vs analytical."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    t_fine = np.linspace(0.0, T_END, 2000)
    z1_anal, z2_anal = analytical_z(t_fine)

    sphere_info = [
        (1, "Sphere 1", z1_anal, r"Mass centre $z_1$ [m]"),
        (2, "Sphere 2", z2_anal, r"Mass centre $z_2$ [m]"),
    ]

    for ax, (col_idx, title, z_anal, ylabel) in zip(axes, sphere_info):
        ax.plot(t_fine, z_anal,
                linewidth=1, linestyle="--", c="k",
                zorder=1, label="Analytical solution")

        for key, (label, color, marker, slc) in INTEGRATORS.items():
            root_name = f"collidingSpheresConv_{FINEST_DT_LABEL}_{key}"
            data = load_data(results_root, root_name)
            t = data[:, 0]
            z = data[:, col_idx]
            ax.plot(t[slc], z[slc],
                    linewidth=0, c=color, marker=marker, markersize=5,
                    markerfacecolor=color, zorder=2, label=label)

        # Put integrator lines first, analytical last
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:] + [handles[0]]
        labels  = labels[1:]  + [labels[0]]
        ax.legend(handles, labels, fontsize=8)

        ax.set_xlabel(r"Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, T_END)
        ax.set_title(title, fontsize=10)
        ax.set_box_aspect(1)
        # Place grid lines beneath plot elements
        ax.set_axisbelow(True)
        ax.grid(color="lightgrey", linestyle="--", linewidth=0.5, zorder=0)

    os.makedirs(plots_root, exist_ok=True)
    fig.tight_layout()
    out = os.path.join(plots_root, "CollidingSpheresConvTrajectory.eps")
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")


# ---------------------------------------------------------------------------
# Convergence plot  (L∞ error vs Δt)
# ---------------------------------------------------------------------------
def plot_convergence(results_root: str, plots_root: str) -> None:
    """Log-log L∞ position error vs dt for both integrators."""
    fig, ax = plt.subplots(figsize=(5, 5))
    dt_arr = np.array(DT_VALUES)
    first_errors: dict = {}

    for key, (label, color, marker, _) in INTEGRATORS.items():
        errors = np.zeros(len(DT_LABELS))
        for i, dt_label in enumerate(DT_LABELS):
            root_name = f"collidingSpheresConv_{dt_label}_{key}"
            data       = load_data(results_root, root_name)
            t, z1_num, z2_num = data[:, 0], data[:, 1], data[:, 2]
            z1_ref, z2_ref    = analytical_z(t)
            e1 = np.max(np.abs(z1_num - z1_ref))
            e2 = np.max(np.abs(z2_num - z2_ref))
            errors[i] = max(e1, e2)
        first_errors[key] = errors[0]
        ax.loglog(dt_arr, errors,
                  linewidth=0, c=color, marker=marker, markersize=7,
                  markerfacecolor=color, zorder=2, label=label)

    # Reference slope lines anchored at the coarsest-dt point
    dt_ref = dt_arr[0]
    ax.loglog(dt_arr, first_errors["FirstOrder"]  * (dt_arr / dt_ref) ** 1,
              "--", linewidth=1, c="k", zorder=1,
              label=r"Slope 1 ($\mathcal{O}(\Delta t)$)")
    ax.loglog(dt_arr, first_errors["SecondOrder"] * (dt_arr / dt_ref) ** 2,
              "-.", linewidth=1, c="k", zorder=1,
              label=r"Slope 2 ($\mathcal{O}(\Delta t^2)$)")

    ax.set_xlabel(r"Time step, $\Delta t$ [s]")
    ax.set_ylabel(r"$L^\infty$ position error [m]")
    ax.set_box_aspect(1)
    ax.legend(fontsize=8)
    # Place grid lines beneath plot elements
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="lightgrey", ls="--", linewidth=0.5, zorder=0)

    os.makedirs(plots_root, exist_ok=True)
    out = os.path.join(plots_root, "CollidingSpheresConvConvergence.eps")
    plt.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trajectory and convergence plots for the collidingSpheresConv test"
    )
    parser.add_argument(
        "--result-dir",
        default="./results/collidingSpheresConv",
        help="Directory containing the *_position_z.dat files",
    )
    parser.add_argument(
        "--plot-dir",
        default="./plots/collidingSpheresConv",
        help="Directory for output .eps plot files",
    )
    args = parser.parse_args()

    print("Colliding Spheres Convergence – physical parameters:")
    print(f"  R         = {R} m")
    print(f"  rho       = {RHO} kg/m³  =>  m = {M:.5f} kg")
    print(f"  kn        = {KN:.1e} N/m  =>  omega = {OMEGA:.4f} rad/s")
    print(f"  g         = {G} m/s²")
    print(f"  V0        = {V0} m/s  (initial velocity of sphere 1)")
    print(f"  t0        = {T0:.4f} s   (free-approach duration until contact)")
    print(f"  v1(t0)    = {V1_AT_T0:.4f} m/s  (sphere 1 velocity at contact)")
    print(f"  v2(t0)    = {V2_AT_T0:.4f} m/s  (sphere 2 velocity at contact)")
    print(f"  Tc        = {T_C * 1e3:.3f} ms  (contact duration)")
    print(f"  z1_end    = {Z1_END:.6f} m   z2_end    = {Z2_END:.6f} m")
    print(f"  v1_post   = {V1_POST:.4f} m/s  v2_post   = {V2_POST:.4f} m/s")
    print()

    plot_trajectory(args.result_dir, args.plot_dir)
    plot_convergence(args.result_dir, args.plot_dir)


if __name__ == "__main__":
    main()
