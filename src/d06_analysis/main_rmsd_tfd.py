import os
import pickle
import matplotlib.pyplot as plt

'''
dict = {
    'name': sys,
    'smiles': smiles,
    'solvent': get_solvent(sys),
    'dielectric': get_dielectric(sys),
    'noe_index': noe_index,
    'ref': ref,
    'bl_etkdg': bl,
    'noe_etkdg': noe_etkdg,
    'omega': omega,
    'vac_min': vac_min,
    'vac_1ns': vac_1ns,
    'vac_5ns': vac_5ns,
    'solv_min': solv_min,
    'ANI_min': ANI_min
    
    bond_dist
    
    'noe_pair_dist'
    bl_pair_dist
    omega_pair_dist
    vac_min_pair_dist
    solv_min_pair_dist
    ANI_min_pair_dist
    
    noe_rb_rmsd
    bl_rb_rmsd
    #omega_rb_rmsd
    vac_min_rb_rmsd
    solv_min_rb_rmsd
    ANI_min_rb_rmsd
    
    noe_tfd
    bl_tfd
    vac_min_tfd
    solv_min_tfd
    ANI_min_tfd
}
'''

os.chdir('../../results/')
data = pickle.load(open('../data/analysis_notfd_gene.data', 'rb'))
'''
for sys in data:
    make_violin_plot_2(sys['noe_index'], sys['bond_dist'], sys['noe_pair_dist'], sys['bl_pair_dist'], sys['name']+'normal')
    make_violin_plot_2(sys['noe_index'], sys['bond_dist'], sys['noe_pair_dist'], sys['vac_min_pair_dist'],
                       sys['name'] + 'vac_min')
    make_violin_plot_2(sys['noe_index'], sys['bond_dist'], sys['noe_pair_dist'], sys['ANI_min_pair_dist'],
                       sys['name'] + 'ANI_min')
'''
for sys in data:
    if sys['dielectric'] > 40:
        sys['marker'] = 'o'
    else:
        sys['marker'] = 'v'


numConf = [1, 10, 50, 54, 100, 540, 5400]

# ETKDG v NOE-ETKDG
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'ETKDGv3'
    y_gen = 'NOE-ETKDG'

    for sys in data:
        x = (min(sys['bl_rb_rmsd'][:n]))
        y = (min(sys['noe_rb_rmsd'][:n]))
        ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()

# ETKDG v NOE-ETKDG
'''
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'ETKDGv3'
    y_gen = 'NOE-ETKDG'

    for sys in data:
        x = (min(sys['bl_tfd'][:n]))
        y = (min(sys['noe_tfd'][:n]))
        ax.scatter(x, y, label=sys['name'])

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'TFD ({x_gen})')
    ax.set_ylabel(f'TFD ({y_gen})')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()
    plt.title(f'n = {n}')
    plt.savefig(f'plots/tfd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()
'''

# NOE-ETKDG v ANI_min
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'ANI Min'

    for sys in data:
        try:
            x = (min(sys['noe_rb_rmsd'][:n]))
            y = (min(sys['ANI_min_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping ANI for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# vac min v ANI_min
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'Vac min'
    y_gen = 'ANI Min'

    for sys in data:
        try:
            x = (min(sys['vac_min_rb_rmsd'][:n]))
            y = (min(sys['ANI_min_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping ANI for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# vac min v Solv
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'Vac min'
    y_gen = 'Solv Min'

    for sys in data:
        try:
            x = (min(sys['vac_min_rb_rmsd'][:n]))
            y = (min(sys['solv_min_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping ANI for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# NOE-ETKDG v vac_min
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'Vac Min'

    for sys in data:
        x = (min(sys['noe_rb_rmsd'][:n]))
        y = (min(sys['vac_min_rb_rmsd'][:n]))
        ax.scatter(10 * x, 10 * y, marker=sys['marker'], label=sys['name'])

    diag_line, = ax.plot((-1, 20), (-1, 20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# NOE-ETKDG v Vac 1ns
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'Vac 1 ns'

    for sys in data:
        try:
            x = (min(sys['noe_rb_rmsd'][:n]))
            y = (min(sys['vac_1ns_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping vac 1ns for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# NOE-ETKDG v Vac 5ns
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'Vac 5 ns'

    for sys in data:
        try:
            x = (min(sys['noe_rb_rmsd'][:n]))
            y = (min(sys['vac_5ns_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping vac 5ns for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()

# Vac 1ns v Vac 5ns
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'Vac 1 ns'
    y_gen = 'Vac 5 ns'

    for sys in data:
        try:
            x = (min(sys['vac_1ns_rb_rmsd'][:n]))
            y = (min(sys['vac_5ns_rb_rmsd'][:n]))
            ax.scatter(10 * x, 10 * y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping vac 5ns for {name}')

    diag_line, = ax.plot((-1, 20), (-1, 20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()

print('d')


# NOE-ETKDG v solv_min
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'Solv Min'

    for sys in data:
        x = (min(sys['noe_rb_rmsd'][:n]))
        y = (min(sys['solv_min_rb_rmsd'][:n]))
        ax.scatter(10 * x, 10 * y, marker=sys['marker'], label=sys['name'])

    diag_line, = ax.plot((-1, 20), (-1, 20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# NOE-ETKDG v Solv 1ns
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'Solv 1 ns'

    for sys in data:
        try:
            x = (min(sys['noe_rb_rmsd'][:n]))
            y = (min(sys['solv_1ns_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping solv 1ns for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


# NOE-ETKDG v Solv 5ns
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'NOE-ETKDG'
    y_gen = 'Solv 5 ns'

    for sys in data:
        try:
            x = (min(sys['noe_rb_rmsd'][:n]))
            y = (min(sys['solv_5ns_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping solv 5ns for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()



# Solv 1ns v Solv 5ns
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'Solv 1 ns'
    y_gen = 'Solv 5 ns'

    for sys in data:
        try:
            x = (min(sys['solv_1ns_rb_rmsd'][:n]))
            y = (min(sys['solv_5ns_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping solv 5ns for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()

# Vac v Solv
for n in numConf:
    fig, ax = plt.subplots(figsize=(6, 6))
    x_gen = 'Vac 1 ns'
    y_gen = 'Solv 1 ns'

    for sys in data:
        try:
            x = (min(sys['vac_1ns_rb_rmsd'][:n]))
            y = (min(sys['solv_1ns_rb_rmsd'][:n]))
            ax.scatter(10*x, 10*y, marker=sys['marker'], label=sys['name'])
        except:
            name = sys['name']
            print(f'skipping md for {name}')

    diag_line, = ax.plot((-1,20), (-1,20), ls="--", c=".3")
    ax.set_xlabel(f'rbRMSD ({x_gen}) / A', fontsize=18)
    ax.set_ylabel(f'rbRMSD ({y_gen}) / A', fontsize=18)
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.legend()
    plt.title(f'n = {n}', fontsize=20)
    plt.savefig(f'plots/rmsd/{x_gen}_v_{y_gen}_{n}.png')
    plt.show()


print('d')
