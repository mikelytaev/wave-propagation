import time

from rwp.SSPade import *
from rwp.WPVis import *
from rwp.environment import *
from rwp.petool import PETOOLPropagationTask

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    env = EarthAtmosphereEnvironment(boundary_condition=SeaSurfaceBC(), height=300)
    ant = GaussSource(wavelength=0.1, height=30, beam_width=2, eval_angle=0, polarz='H')
    max_range = 150000
    pade12_task = SSPadePropagationTask(src=ant, env=env, two_way=False, max_range_m=max_range, pade_order=(7, 8),
                                        dx_wl=400, n_dx_out=1, dz_wl=1, n_dz_out=1)
    pade12_field = pade12_task.calculate()

    petool_task = PETOOLPropagationTask(src=ant, env=env, two_way=False, max_range_m=max_range, dx_wl=400, n_dx_out=1,
                                        dz_wl=3)
    petool_field = petool_task.calculate()

    pade12_vis = FieldVisualiser(pade12_field, trans_func=lambda v: 10 * cm.log10(1e-16 + abs(v)),
                                 label='Pade-[7/8] + NLBC', x_mult=1E-3)
    petool_vis = FieldVisualiser(petool_field, trans_func=lambda x: x, label='SSF (PETOOL)', x_mult=1E-3)
    plt = petool_vis.plot_hor(30, pade12_vis)
    plt.xlabel('Range (km)')
    plt.ylabel('10log|u| (dB)')
    plt.tight_layout()
    plt.show()

    plt = petool_vis.plot2d(min=-120, max=0)
    plt.title('10log|u|')
    plt.xlabel('Range (km)')
    plt.ylabel('Height (m)')
    plt.tight_layout()
    plt.show()
    #plt.savefig("std_2dpetool.eps", dpi='figure')

    plt = pade12_vis.plot2d(min=-120, max=0)
    plt.title('10log|u|')
    plt.xlabel('Range (km)')
    plt.ylabel('Height (m)')
    plt.tight_layout()
    plt.show()
    #plt.savefig("std_2d78.eps", dpi='figure')