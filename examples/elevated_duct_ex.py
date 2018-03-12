import time

from rwp.SSPade import *
from rwp.WPVis import *
from rwp.environment import *
from rwp.petool import PETOOLPropagationTask
import matplotlib

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    profile1d = interp1d(x=[0, 100, 150, 300], y=[0, 32, 10, 50], fill_value="extrapolate")
    def profile2d(x, z):
        return profile1d(z)
    env = EarthAtmosphereEnvironment(boundary_condition=PECSurfaceBC(), height=300, M_profile=profile2d)
    env.terrain = KnifeEdges([70000], [150])
    ant = GaussSource(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
    max_range = 100000
    pade12_task = SSPadePropagationTask(src=ant, env=env, two_way=True, max_range_m=max_range, pade_order=(7, 8),
                                        dx_wl=400, n_dx_out=1, dz_wl=1, n_dz_out=1)
    pade12_field = pade12_task.calculate()

    petool_task = PETOOLPropagationTask(src=ant, env=env, two_way=True, max_range_m=max_range, dx_wl=400, n_dx_out=1, dz_wl=3)
    petool_field = petool_task.calculate()

    matplotlib.rcParams.update({'font.size': 10})

    pade12_vis = FieldVisualiser(pade12_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                 label='Pade-[7/8] + NLBC', x_mult=1E-3)
    petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
    plt = pade12_vis.plot_hor(150, petool_vis)
    plt.xlabel('Range (km)')
    plt.ylabel('10log|u| (dB)')
    plt.tight_layout()
    plt.show()
    #plt.savefig("elevated_hor.eps")

    plt = petool_vis.plot2d(min=-70, max=0)
    plt.xlabel('Range (km)')
    plt.ylabel('Height (m)')
    plt.tight_layout()
    plt.show()

    plt = pade12_vis.plot2d(min=-70, max=0)
    plt.title('10log|u|')
    plt.xlabel('Range (km)')
    plt.ylabel('Height (m)')
    plt.tight_layout()
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    pade12_vis.plot_ver(50*1E3, ax1, petool_vis)
    ax1.set_ylabel('Height (m)')
    ax1.set_xlabel('10log|u| (dB)')

    pade12_vis.plot_ver(80 * 1E3, ax2, petool_vis)
    ax2.set_ylabel('Height (m)')
    ax2.set_xlabel('10log|u| (dB)')
    f.tight_layout()
    f.show()
    # plt.savefig("elevated_2d.eps")

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # ax1.plot([a / 6371000 * 1E6 for a in pade12_field.z_grid], pade12_field.z_grid)
    # ax1.legend()
    # ax1.set_xlabel('M-units')
    # ax1.set_ylabel('Height (m)')
    #
    # ax2.plot([profile1d(a) for a in pade12_field.z_grid], pade12_field.z_grid)
    # ax2.legend()
    # ax2.set_xlabel('M-units')
    # ax2.set_ylabel('Height (m)')
    # f.tight_layout()
    # f.show()