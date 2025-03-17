import numba

@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)

def maximum_path_jit(paths, values, t_ys, t_xs):
    max_neg_val = -1e9
    for i in range(paths.shape[0]):
        path, value, t_y, t_x = paths[i], values[i], t_ys[i], t_xs[i]
        
        # Forward pass
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_prev = max_neg_val if x == 0 and y > 0 else 0.0 if x == 0 and y == 0 else value[y - 1, x - 1]
                v_cur = max_neg_val if x == y else value[y - 1, x]
                value[y, x] += max(v_prev, v_cur)
        
        # Backtracking
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index -= 1