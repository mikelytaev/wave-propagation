from rwp.environment import *
from rwp.field import *
from rwp.vis import *


def read_winprop_ascii(filename):
    with open(filename) as f:
        content = f.read().splitlines()

    header = True
    for line in content:
        if line == 'END_DATA':
            break
        if header:
            if line == 'BEGIN_DATA':
                header = False
                field = np.empty((lines, columns), dtype=complex)
                continue
            line_arr = line.split(' ')
            if line_arr[1] == 'COLUMNS':
                columns = int(line_arr[2])
            if line_arr[1] == 'LINES':
                lines = int(line_arr[2])
            if line_arr[1] == 'FREQUENCY':
                freq_mhz = float(line_arr[2])
            if line_arr[1] == 'HEIGHT':
                height_m = float(line_arr[2])
            if line_arr[1] == 'RESOLUTION':
                resolution_m = float(line_arr[2])
        else:
            line_arr = line.split(' ')
            x = float(line_arr[0])
            y = float(line_arr[1])
            column = round((x - 1) / resolution_m)
            row = round((y - 1) / resolution_m)
            val = float(line_arr[2])
            field[column, row] = val

    x_grid = np.arange(0, columns - 1) * resolution_m + 1
    y_grid = np.arange(0, lines - 1) * resolution_m + 1
    z_grid = np.array([height_m])
    f = Field3d(x_grid, y_grid, z_grid)
    f.field[:, :, 0] = field[:-1:, :-1:]

    return f



filename = 'C:\\Users\\Mikhail\\Desktop\\feko_propagation\\paper\\free_space\\unnamed1.txt'
f = read_winprop_ascii(filename)


rt_vis = FieldVisualiser3D(field=f, trans_func=lambda v: v, label='3D Ray-tracing')

rt_vis.plot_xy(z0=1.5, min_val=-96, max_val=-26)
plt.xlabel('Расстояние, м')
plt.ylabel('y, м)')
plt.tight_layout()
plt.show()

rt_vis.plot_x(y0=350, z0=1.5)
plt.xlabel('Расстояние, м')
plt.ylabel('Power (dBm)')
plt.grid(True)
plt.tight_layout()
plt.show()