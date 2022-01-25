def get_mean_std(INPUT_SIZE, bench_name):
    mean, std = None, None

    if INPUT_SIZE == 128:
        if bench_name == 'shanghai_A':
            mean = [0.32242125, 0.3308784, 0.36531266]
            std = [0.20520084, 0.20671864, 0.21981789]
        elif bench_name == 'shanghai_B':
            mean = [0.42691906, 0.44187748, 0.4461924]
            std = [0.16749409, 0.17348416, 0.18042755]

    elif INPUT_SIZE == 256:
        if bench_name == 'shanghai_A':
            mean = [0.2983559, 0.3062046, 0.33802828]
            std = [0.21958432, 0.22060138, 0.23461229]
        elif bench_name == 'shanghai_B':
            mean = [0.42692594, 0.44189072, 0.44618444]
            std = [0.18650701, 0.19225413, 0.19943645]
        elif bench_name == 'ucf_18_hajj':  # TODO: recalculate
            mean = [0.38783104, 0.38375905, 0.38995425]
            std = [0.38323926, 0.39154493, 0.39075972]
        elif bench_name == 'ucf_18':
            mean = [0.33189618, 0.34023062, 0.3728201]
            std = [0.16668856, 0.17049365, 0.18432766]
        elif bench_name == 'ucf_18v1':
            mean = [0.33024199, 0.33794826, 0.36828972]
            std = [0.1767343, 0.18027293, 0.19388486]

    elif INPUT_SIZE == 512:
        if bench_name == 'shanghai_A':
            mean = [0.23139508, 0.2374646, 0.26217736]
            std = [0.22035383, 0.22130816, 0.23672004]
        elif bench_name == 'shanghai_B':
            mean = [0.32018929, 0.33140811, 0.3346443]
            std = [0.23334228, 0.23878472, 0.24407709]
        elif bench_name == 'ucf_18':
            mean = [0.30767617, 0.31538381, 0.34562042]
            std = [0.18573873, 0.18862324, 0.20404656]
        elif bench_name == 'ucf_18v1':
            mean = [0.28796011, 0.29466529, 0.32114294]
            std = [0.18639777, 0.18898598, 0.20312717]
        elif bench_name == 'nwpu':
            mean = [0.33574451, 0.3482778, 0.37829334]
            std = [0.20596884, 0.2087179, 0.22302162]
        elif bench_name == 'nwpu_v1':
            mean = [0.30298058, 0.31315339, 0.34048589]
            std = [0.18599708, 0.18967585, 0.20333949]

    return mean, std
