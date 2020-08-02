from collections import defaultdict
import copy
import sys

from scipy.spatial.transform import Rotation as R
import numpy
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
plt.style.use('seaborn-pastel')


fps = 30 # 30
length = 60 # seconds
max_number_of_frames = length * fps
bitrate = 1800
dpi = 100 # 100
width = 1280 / dpi
height = 720 / dpi

fig = plt.figure(figsize=(width, height))
fig.tight_layout()
ax = plt.axes(xlim=(0.5, 4.5), ylim=(0, 1))
plt.gca().set_position([0, 0, 1, 1])
# ax.set_adjustable("datalim")
# line, = ax.plot([], [], lw=3)
scatter_solid = ax.scatter([], [], alpha=1)
scatter_alpha = ax.scatter([], [], alpha=0.3)
scatter_min_diff = ax.scatter([], [], alpha=0.3)
scatter_max_diff = ax.scatter([], [], alpha=0.3)
scatter_bootstrap = ax.scatter([], [], alpha=0.3)
scatter_dataorder = ax.scatter([], [], alpha=0.3)
scatter_dropout = ax.scatter([], [], alpha=0.3)
scatter_random_search = ax.scatter([], [], alpha=0.3)
scatter_bayesopt = ax.scatter([], [], alpha=0.3)

noise_scatters = [scatter_alpha, scatter_bootstrap, scatter_dataorder, scatter_dropout,
                  scatter_random_search, scatter_bayesopt]

plt.axis('off')


N_INTRO = 100
N_TYPICAL = 50 + N_INTRO
N_STOCHASTIC = 100 + N_TYPICAL
N_SWAPING = 200 + N_STOCHASTIC
N_TYPICAL_VAR = 200 + N_SWAPING
N_MULTIPLE_VAR = 250 + N_TYPICAL_VAR
N_VARIANCE_STUDY = 150 + N_MULTIPLE_VAR
N_INITS = 100 + N_VARIANCE_STUDY
N_INIT_SELECTION = 25 + N_INITS
N_DROP_SELECTION = 75 + N_INIT_SELECTION
N_ASSUMED_DIFF = 100 + N_DROP_SELECTION
N_REGRESSION_MEAN = 50 + N_ASSUMED_DIFF
N_RESULTS = 50 + N_REGRESSION_MEAN
N_POINTS = N_STOCHASTIC - N_TYPICAL


rng = numpy.random.RandomState(1)



def init():
    line.set_data([], [])
    return line,

def animate(i):
    if i < N_INTRO:
        cover(i)
    elif i < N_TYPICAL:
        typical_benchmark(i - N_INTRO)
    elif i < N_STOCHASTIC:
        stochastic_benchmark(i - N_TYPICAL)
    elif i < N_SWAPING:
        swaping_rankings(i - N_STOCHASTIC)
    elif i < N_TYPICAL_VAR:
        typical_variance(i - N_SWAPING)
    elif i < N_MULTIPLE_VAR:
        multiple_variances(i - N_TYPICAL_VAR)
    elif i < N_VARIANCE_STUDY:
        variance_study(i - N_MULTIPLE_VAR)
    elif i < N_INITS:
        inits(i - N_VARIANCE_STUDY)
    elif i < N_INIT_SELECTION:
        init_selection(i - N_INITS)
    elif i < N_DROP_SELECTION:
        drop_selection(i - N_INIT_SELECTION)
    elif i < N_ASSUMED_DIFF:
        assumed_diff(i - N_DROP_SELECTION)
    elif i < N_REGRESSION_MEAN:
        regression_mean(i - N_ASSUMED_DIFF)
    elif i < N_RESULTS:
        results(i - N_REGRESSION_MEAN)

    return scatter_solid,


MODELS = 'ABCD'
model_xs = [1.3, 2.2, 3.1, 4]
base = dict(A=0.8, B=0.77, C=0.73, D=0.72)
mean = dict(A=0.77, B=0.75, C=0.75, D=0.74)
colors = dict((model, i / len(MODELS)) for i, model in enumerate(MODELS))
color_array = numpy.array([colors[model] for model in MODELS])

LABEL_SIZE = 60

algorithms_positon = (0.5, 0.05)
algorithms_text = plt.text(
    -1, -1, 'Algorithms', transform=plt.gcf().transFigure,
    fontsize=LABEL_SIZE, horizontalalignment='center', verticalalignment='bottom')

performance_position = (0.1, 0.65)
performance_text = plt.text(
    -1, -1, 'Performances',
    transform=plt.gcf().transFigure, fontsize=50, horizontalalignment='center',
    verticalalignment='center', rotation=90)

best_text = plt.text(
    0, 0, 'Best', fontsize=20, horizontalalignment='center',
    verticalalignment='center',)

worst_text = plt.text(
    0, 0, 'Worst', fontsize=20, horizontalalignment='center',
    verticalalignment='center',)

bootstrap_text = plt.text(
    0, 0, 'Bootstrap',
    fontsize=30, horizontalalignment='right',
    verticalalignment='center')


weight_inits_text = plt.text(
    0, 0, 'Weights init',
    fontsize=30, horizontalalignment='right',
    verticalalignment='center')


ordering_text = plt.text(
    0, 0, 'Data order',
    fontsize=30, horizontalalignment='right',
    verticalalignment='center')


dropout_text = plt.text(
    0, 0, 'Dropout',
    fontsize=30, horizontalalignment='right',
    verticalalignment='center')


random_search_text = plt.text(
    0, 0, 'Random Search',
    fontsize=30, horizontalalignment='right',
    verticalalignment='center')


bayesopt_text = plt.text(
    0, 0, 'Bayes Opt',
    fontsize=30, horizontalalignment='right',
    verticalalignment='center')


noise_texts = [weight_inits_text, bootstrap_text, ordering_text, dropout_text,
               random_search_text, bayesopt_text]


task_texts = []
for task in ['SST-2\nBERT', 'RTE\nBERT', 'PascalVOC\nResNet', 'CIFAR10\nVGG11', 'MHC\nMLP']:
    task_texts.append(plt.text(
        0, 0, task,  # task.replace('\n', '-'),
        fontsize=20, horizontalalignment='center',
        verticalalignment='center'))


noise_text_x = 1.6

noise_text_ys = [
    0.45, # init
    0.55, # bootstrap
    0.35,  # dataorder
    0.25, # dropout
    0.65,  # random search
    0.75 # bayes opt
    ]

legends_positions = dict(
    (model, (model_xs[j], 0.2)) for j, model in enumerate(MODELS))

legends = [plt.text(-1, -1, model, fontsize=50,
                    fontweight='bold', horizontalalignment='center')
           for j, model in enumerate(MODELS)]


title = plt.text(
    0.5, 0.6,
    ('Simulated Hyperparameter Optimization\n'
     'for Statistical Tests in Machine Learning Benchmarks'),
    fontsize=30, horizontalalignment='center', transform=plt.gcf().transFigure,
    verticalalignment='center')


black_patch = patches.Rectangle((0, 0), 0, 0, fill=True, color='black')
ax.add_patch(black_patch)


def cover(i):
    # TODO Add title
    # TODO Add authors
    # TODO Add logos

    if i > N_INTRO - 10:
        black_patch.set_width(5)
        black_patch.set_height(5)
        title.set_position((-1, -1))


def typical_benchmark(i):
    black_patch.set_width(0)
    black_patch.set_height(0)

    performance_text.set_position(performance_position)
    algorithms_text.set_position(algorithms_positon)

    for model, legend in zip(MODELS, legends):
        legend.set_position(legends_positions[model])

    A = (1, 0.8)
    B = (2, 0.75)
    C = (3, 0.73)
    D = (4, 0.7)

    scatter_solid.set_offsets([(model_xs[j], base[model]) for j, model in enumerate('ABCD')])
    scatter_solid.set_sizes([2000 for model in MODELS])
    scatter_solid.set_array(color_array)


cloud = defaultdict(list)

data = numpy.zeros((len(MODELS), 2, 500))
for j, model in enumerate('ABCD'):
    data[j, 0, :] = rng.normal(model_xs[j], 0.02, size=500)
    data[j, 1, :] = rng.normal(mean[model], 0.1, size=500)

data_vars = numpy.zeros((6, 5, 100, 2))
types = ['weights_init', 'bootstrap', 'dataorder', 'dropout', 'random_search', 'bayesopt']
noise_xs = [
    5.3,
    11.3,
    16.3,
    21.3,
    26.3]
noise_ys = [
    0.45, # init 
    0.54, # bootstrap
    0.35, # dataorder
    0.25, # dropout
    0.65, # random search
    0.75]  # bayesopt
task_ys = [
    0.80,
    0.65,
    0.50,
    0.35,
    0.2]
vgg = [
    0.0023054717521583286,
    0.0035756520733986412,
    0.002744239011092143,
    0,
    0.0022843520673959546,
    0.0044826302194370705]

bert_sst2 = [
    0.002344054700882457,
    0.007684901394424942,
    0.0030550343764080483,
    0.0031318425973985643,
    0.0030134882249386835,
    0.003416461683823229]

DEMO_TASK = 0  # SST2
SST2 = 0
RTE = 1
SEG = 2
VGG = 3
MLP = 4
for noise_type in range(len(types)):
    data_vars[noise_type, 0, :, 0] =  rng.normal(0, bert_sst2[noise_type] * 100, size=N_POINTS)
    data_vars[noise_type, 0, :, 1] =  rng.normal(0, 0.01, size=N_POINTS)

    data_vars[noise_type, 1, :, 0] =  rng.normal(0, bert_sst2[noise_type] * 100, size=N_POINTS)
    data_vars[noise_type, 1, :, 1] =  rng.normal(0, 0.01, size=N_POINTS)

    data_vars[noise_type, 2, :, 0] =  rng.normal(0, bert_sst2[noise_type] * 100, size=N_POINTS)
    data_vars[noise_type, 2, :, 1] =  rng.normal(0, 0.01, size=N_POINTS)

    data_vars[noise_type, 3, :, 0] =  rng.normal(0, vgg[noise_type] * 100, size=N_POINTS)
    data_vars[noise_type, 3, :, 1] =  rng.normal(0, 0.01, size=N_POINTS)

    data_vars[noise_type, 4, :, 0] =  rng.normal(0, bert_sst2[noise_type] * 100, size=N_POINTS)
    data_vars[noise_type, 4, :, 1] =  rng.normal(0, 0.01, size=N_POINTS)

data_original = copy.deepcopy(data_vars)


# segmentation
# bootstrapping_seed 0.012707861420580065
# init_seed 0.005112707058290076
# sampler_seed 0.0035452443972458845
# reference 0.0016047613674296823
# noisy_grid_search 0.002589220496953402
# random_search 0.007181016531841449
# bayesopt 0.002303471392478707
# Computing variances for bert-sst2
# 20
# 50
# bert-rte
# bootstrapping_seed 0.03065957140128678
# global_seed 0.013886475279050828
# init_seed 0.02698042765396676
# sampler_seed 0.016432685985090543
# noisy_grid_search 0.01740020536710826
# random_search 0.022089133583104853
# bayesopt 0.019071834730117608
# Computing variances for bio-task2
# 20
# bio-task2
# bootstrap_seed 0.03075609838422315
# random_state 0.01677507919822131
# noisy_grid_search 0.01623419961679567
# random_search 0.012599712304922458
# bayesopt 0.016308436481642236


colors_matrix = numpy.ones((len(MODELS), 500)) * color_array[:, None]


def stochastic_benchmark(i):
    A = (1, 0.8)
    B = (2, 0.5)
    C = (3, 0.3)
    D = (4, 0.2)
    
    n_points = min(i + 1, N_POINTS)
    data_i = data[:, :, :n_points].transpose(0, 2, 1).reshape(-1, 2)
    scatter_alpha.set_offsets(data_i)
    scatter_alpha.set_sizes(numpy.ones(n_points) * 300)
    scatter_alpha.set_array(colors_matrix[:, :n_points].reshape(-1))


def swaping_rankings(i):
    swap_rankings(i, i // 50, i % 50, 50)

def swap_rankings(ri, k, i, t):

    def _get_prev_rank(k, j):
        if k == 0:
            return j
        
        return idx.index(j)

    # Find rankings

    # Swap the positions
    # select k-th points as the centered one


    if k == 0:
        prev_idx = list(range(len(MODELS)))
    else:
        data_i = data[:, :, k - 1]
        prev_idx = list(numpy.argsort(data_i[:, 1])[::-1])

    # data_alpha = copy.deepcopy(data)

    saturation = 10

    data_i = copy.deepcopy(data[:, :, k])
    data_tmp = copy.deepcopy(data)
    idx = numpy.argsort(data_i[:, 1])[::-1]
    for rank, j in enumerate(idx):
        prev_rank = prev_idx.index(j)
        model = MODELS[j]
        delta = (sigmoid(i / t * saturation) - 0.5) * 2 * (model_xs[rank] - model_xs[prev_rank])
        position = delta + model_xs[prev_rank]
        data_i[j, 0] = position
        # data[j, 0, :] = position
        legends_positions[MODELS[j]] = (position, 0.2)
        legends[j].set_position((position, 0.2))
        # data[j, 0, :] += ((rank - prev_rank) / t)
        delta = ((sigmoid(i / t * saturation) - 0.5) * 2 -
                 (sigmoid((i - 1) / t * saturation) - 0.5) * 2)
        data_tmp[j, 0, :] = position + (data_tmp[j, 0, :] - data_tmp[j, 0, :].mean())

    scatter_solid.set_offsets(data_i.reshape(-1, 2))
    # scatter_alpha.set_sizes(numpy.ones(i) * 300)
    # scatter_alpha.set_array(colors_matrix[:, :i].reshape(-1))
    data_i = data_tmp[:, :, :N_POINTS].transpose(0, 2, 1).reshape(-1, 2)
    scatter_alpha.set_offsets(data_i)
    # scatter_alpha.set_sizes(numpy.ones(i) * 300)
    # scatter_alpha.set_array(colors_matrix[:, :i].reshape(-1))

    if ri == (N_SWAPING - N_STOCHASTIC - 1):
        data[:, 0, :] = data_tmp[:, 0, :]


def sigmoid(t):
    return 1 / (1 + numpy.exp(-t))


def typical_variance(i):

    # Find rankings

    # Swap the positions
    # select k-th points as the centered one

    # data_alpha = copy.deepcopy(data)

    saturation = 10

    mean_x = 0.02
    var_x = 0.1
    mean_y = -0.01
    var_y = 0.05

    speed = 0.05

    offsets = scatter_solid.get_offsets()

    offsets[:, 0] += rng.normal((offsets[:, 0] - 2.5) * speed, var_x, size=len(MODELS))
    offsets[:, 1] += rng.normal((offsets[:, 1] - 0.5) * speed, var_y, size=len(MODELS))
    scatter_solid.set_offsets(offsets.reshape(-1, 2))

    data[1:, 0, :] += rng.normal(
        (data[1:, 0, :] - 2.5) * speed,
        var_x,
        size=(len(MODELS) - 1, data.shape[2]))
    data[1:, 1, :] += rng.normal(
        (data[1:, 1, :] - 0.5) * speed,
        var_y,
        size=(len(MODELS) - 1, data.shape[2]))

    data_i = data[:, :, :N_POINTS].transpose(0, 2, 1).reshape(-1, 2)
    # Don't save rotation.
    rotated = copy.deepcopy(data)

    origin = numpy.array((
        model_xs[1],
        0.5,
        0))[None, :]
    new_origin = numpy.array((
        translate(2, 0.6, i, 50),
        translate(0.5, 0.9, i, 50),
        0))[None, :]

    scale = numpy.array((
        translate(1, 3, i, 50), 
        translate(1, 0.5, i, 50),
        1))[None, :]
    rotation = translate(0, -90, i, 50)
    r = R.from_euler('z', rotation, degrees=True)
    xy = data[0, :, :N_POINTS]
    xyz = numpy.concatenate([xy, numpy.ones((1, xy.shape[1]))], axis=0).T
    xyz = (r.apply(xyz - origin) + new_origin) * scale

    rotated[0, :, :N_POINTS] = xyz[:, :2].T
    data_i = rotated[:, :, :N_POINTS].transpose(0, 2, 1).reshape(-1, 2)
    scatter_alpha.set_offsets(data_i.reshape(-1, 2))
    for j, model in enumerate(MODELS):
        position = numpy.array(legends_positions[model])
        position[0] += rng.normal((position[0] - 2) * 0.1, var_x)
        position[1] += rng.normal((position[1] - 0.5) * 0.1, var_y)
        legends_positions[model] = position
        legends[j].set_position(position)
    # scatter_alpha.set_sizes(numpy.ones(i) * 300)
    # scatter_alpha.set_array(colors_matrix[:, :i].reshape(-1))
    # scatter_alpha.set_offsets(data_i)
    # scatter_alpha.set_sizes(numpy.ones(i) * 300)
    # scatter_alpha.set_array(colors_matrix[:, :i].reshape(-1))

    size = translate(2000, 0, i, 100)
    scatter_solid.set_sizes([size for _ in MODELS])
    size = translate(300, 0, i, 100)
    sizes = numpy.ones((len(MODELS), N_POINTS)) * size
    sizes[0, :] = 300
    scatter_alpha.set_sizes(sizes.reshape(-1))

    # Drop Algorithms label
    position = algorithms_text.get_position()
    algorithms_text.set_position((position[0], position[1] - 0.02))

    # Rotate performances label
    # rotation = performance_text.get_rotation()
    performance_text.set_rotation(translate(90, 0, i, 50))
    # position = performance_text.get_position()
    # performance_text.set_position((position[0] + 0.05, position[1] - 0.02))
    position = (
        translate(performance_position[0], 0.45, i, 50),
        translate(performance_position[1], 0.075, i, 50))
    performance_text.set_position(position)

    position = (
        translate(0, noise_text_x, i, 50), noise_text_ys[0])
    weight_inits_text.set_position(position)


def translate(a, b, step, steps, saturation=10):
    return a + (sigmoid(step / steps * saturation) - 0.5) * 2 * (b - a)


duration = 25

def multiple_variances(i):
    for j, noise_type in enumerate(types[1:]):
        if i < (j + 1) * duration:
            noise_cloud(i - j * duration, noise_type)
            break

    # if i < duration:
    #     noise_cloud(i, )
    # elif i < 2 * duration:
    #     noise_cloud(i - duration, 'dataorder')
    # elif i < 3 * duration:
    #     noise_cloud(i - 2 * duration, 'dropout')
    # elif i < 4 * duration:
    #     random_search(i - 3 * duration)
    # elif i < 5 * duration:
    #     bayesopt(i - 4 * duration)


def noise_cloud(i, noise_type):
    index = types.index(noise_type)
    position = (translate(0, noise_text_x, i, duration), noise_text_ys[index])
    scatter_noise = noise_scatters[index]
    noise_texts[index].set_position(position)

    data_i = data_vars[index, DEMO_TASK, :(i + 1) * N_POINTS // duration]
    data_i = copy.deepcopy(data_i)
    data_i[:, 0] += 2.5
    data_i[:, 1] += noise_ys[index]
    scatter_noise.set_offsets(data_i)
    scatter_noise.set_sizes(numpy.ones(data_i.shape[0]) * 300)



VARIANCE_REDUCE = 8

def variance_study(i):
    # for text_artist in [bayesopt_text, random_search_text, bootstrap_text, weight_inits_text,
    #                     ordering_text, dropout_text]:
    #     position = (
    #         translate(0.3, 0.18, i, 50),
    #         text_artist.get_position()[1])
    #     text_artist.set_position(position)
    #     text_artist.set_fontsize(translate(30, 20, i, 50))

    online_data = dict()
    online_size = dict()
    for noise_type, y in zip(types, noise_ys):
        data_i = data_vars[types.index(noise_type), DEMO_TASK, :N_POINTS]
        data_i = copy.deepcopy(data_i)
        data_i[:, 1] /= translate(1, 5, i, 50)
        data_i[:, 1] += y
        mean = data_i[:, 0].mean()
        data_i[:, 0] -= mean
        print('mean', mean)
        # data_i[:, 0] /= max(min(i / 50, 1) * VARIANCE_REDUCE, 1)  # translate(1, VARIANCE_REDUCE, i, 50)
        data_i[:, 0] /= translate(1, VARIANCE_REDUCE, i, 50)
        data_i[:, 0] += translate(mean + 2.5, 1.78, i, 50)
        online_data[noise_type] = data_i
        online_size[noise_type] = numpy.ones(data_i.shape[0]) * translate(300, 100, i, 50) 

    scatter_alpha.set_array(numpy.ones(N_POINTS * 5) * color_array[0])

    task_texts[DEMO_TASK].set_position((2.3 / VARIANCE_REDUCE + 1.5, translate(1, 0.9, i, 50)))

    if i > 25:
        other_task(i - 25, online_data, online_size, RTE, 11.3)
    if i > 50:
        other_task(i - 50, online_data, online_size, SEG, 16.3)
    if i > 75:
        other_task(i - 75, online_data, online_size, VGG, 21.3)
    if i > 100:
        other_task(i - 100, online_data, online_size, MLP, 26.3)

    for noise_type, scatter_noise in zip(types, noise_scatters):
        scatter_noise.set_offsets(online_data[noise_type])
        scatter_noise.set_sizes(online_size[noise_type])


def other_task(i, online_data, online_size, task, x):

    for noise_type, y in zip(types, noise_ys):
        data_i = data_vars[types.index(noise_type), task, :(i + 1) * N_POINTS // duration]
        data_i = copy.deepcopy(data_i)
        data_i[:, 0] += x
        data_i[:, 1] /= 5
        data_i[:, 1] += y
        data_i[:, 0] /= VARIANCE_REDUCE
        data_i[:, 0] += 1
        online_data[noise_type] = numpy.concatenate([online_data[noise_type], data_i], axis=0)
        sizes = numpy.ones(data_i.shape[0]) * translate(300, 100, i, 50)
        online_size[noise_type] = numpy.concatenate([online_size[noise_type], sizes], axis=0)

    # TODO: Place labels correctly
    task_texts[task].set_position((x / VARIANCE_REDUCE + 1, translate(1, 0.9, i, 50)))


def inits(i):

    var_x = 0.05
    var_y = 0.01

    if i == 0:
        for j in range(5):
            for noise_type, y in zip(types, noise_ys):
                index = types.index(noise_type)
                data_vars[index, j, :, 0] += noise_xs[j]
                data_vars[index, j, :, 1] /= 5
                data_vars[index, j, :, 1] += y
                data_vars[index, j, :, 0] /= VARIANCE_REDUCE
                data_vars[index, j, :, 0] += 1
                data_original[index, j] = data_vars[index, j]

    for noise_type, scatter_noise in zip(types[1:], noise_scatters[1:]):
        data_i = data_vars[types.index(noise_type), :, :N_POINTS]
        data_i = data_i.reshape((-1, 2))
        data_i[:, 0] += rng.normal((data_i[:, 0] - data_i[:, 0].mean()) * 0.05, var_x, size=N_POINTS * 5)
        data_i[:, 1] += rng.normal((data_i[:, 1] - data_i[:, 1].mean()) * 0.1, var_y, size=N_POINTS * 5)

        scatter_noise.set_offsets(data_i)
        scatter_noise.set_sizes([translate(100, 0, i, 100) for _ in range(N_POINTS * 5)])

    # TODO: Remove noise labels (but move up weights init)
    # TODO: Place labels correctly
    for noise_text in noise_texts[1:]:
        position = noise_text.get_position()
        x = position[0] + rng.normal((position[0] - 2) * 0.1, var_x)
        y = position[1] + rng.normal((position[1] - 0.5) * 0.05, var_y)
        noise_text.set_position((x, y))

    weight_inits_text.set_position((noise_text_x, translate(noise_text_ys[0], 0.93, i, 100)))

    for task in range(5):
        task_texts[task].set_horizontalalignment('right')
        task_texts[task].set_verticalalignment('center')
        task_texts[task].set_position((
            translate(noise_xs[task] / VARIANCE_REDUCE + 1, 1, i, 100),
            translate(0.9, task_ys[task], i, 100)))

    # TODO: Realign weights init
    # TODO: The data is task specific, but we must combine it in the same scatter. 
    #       We must append all data to a single array, and then call scatter.set_offsets.
    data = numpy.zeros((N_POINTS * 5, 2))
    for task in range(5):
        data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
        # data_i = data_vars[types.index('weights_init'), task, :N_POINTS]
        data[task * N_POINTS:(task + 1) * N_POINTS, 0] = translate(
            data_original_i[:, 0],
            (2.5 + (data_original_i[:, 0] - data_original_i[:, 0].mean()) * VARIANCE_REDUCE),
            i, 100)
        data[task * N_POINTS:(task + 1) * N_POINTS, 1] = translate(
            data_original_i[:, 1],
            task_ys[task] + data_original_i[:, 1] - data_original_i[:, 1].mean(),
            i, 100)
        # data[] = data_i
    scatter_alpha.set_offsets(data)
    scatter_alpha.set_sizes([100 for _ in range(N_POINTS * 2)])


def init_selection(i):
    # Drop all points other than best and worst.
    # Maybe make it like the others are falling down...
    # And increase size of selection ones, maybe

    duration = 25

    if i == 0:
        for task in range(5):
            data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
            data_original_i[:, 0] = (
                2.5 + (data_original_i[:, 0] - data_original_i[:, 0].mean()) * VARIANCE_REDUCE)
            data_original_i[:, 1] = task_ys[task] + data_original_i[:, 1] - data_original_i[:, 1].mean()

    best_text.set_position((2, translate(1, 0.85, i, duration)))

    delay = 15

    if i > delay:
        worst_text.set_position((3, translate(1, 0.85, i - delay, duration)))

    def get_size(j, idx):
        if j == idx[0]:
            return translate(100, 500, i, duration)
        elif j == idx[-1] and i > delay:
            return translate(100, 500, i - delay, duration)
        else:
            return 100

    sizes = numpy.zeros(N_POINTS * 5)
    for task in range(5):
        data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
        idx = numpy.argsort(data_original_i[:, 0])
        sizes[task * N_POINTS:(task + 1) * N_POINTS] = [get_size(j, idx) for j in range(N_POINTS)]

    scatter_alpha.set_sizes(sizes)


def drop_selection(i):
    
    i += 15  # Otherwise the start is too slow

    data = numpy.zeros((N_POINTS * 5, 2))
    min_x = data_original[types.index('weights_init'), :, :N_POINTS, 0].min()
    max_x = data_original[types.index('weights_init'), :, :N_POINTS, 0].max()
    delta = max_x - min_x
    for task in range(5):
        data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
        task_min_x = data_original_i[:, 0].min()
        task_max_x = data_original_i[:, 0].max()
        threshold = max(min((i / 25) * delta, task_max_x), task_min_x)
        mask = (task_min_x < data_original_i[:, 0]) * (data_original_i[:, 0] < threshold)
        data_original_i[:, 1] = data_original_i[:, 1] - mask * 1 / rng.normal(20, 5, size=N_POINTS)
        data[task * N_POINTS:(task + 1) * N_POINTS] = data_original_i

    scatter_alpha.set_offsets(data)


assumed_data_min = numpy.zeros((N_POINTS, 5, 2))
assumed_data_max = numpy.zeros((N_POINTS, 5, 2))

def assumed_diff(i):
    n_points = N_POINTS // 50
    data_min = numpy.zeros(((i + 1) * n_points * 5, 2))
    data_max = numpy.zeros(((i + 1) * n_points * 5, 2))

    if (i + 1) * n_points > N_POINTS:
        return

    def sample(task, task_extremum, original, memory, data_buffer):
        # TODO: Take actual variance of the task (global variance, not only weights init)
        memory[i * n_points:(i + 1) * n_points, task, 0] = rng.normal(
            task_extremum, 0.2, size=n_points)

        memory[i * n_points:(i + 1) * n_points, task, 1] = rng.normal(
            task_ys[task], 0.005, size=n_points)

        data_buffer[task * (i + 1) * n_points: (task + 1) * (i + 1) * n_points] = memory[:(i + 1) * n_points, task]


    for task in range(5):
        data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
        task_min_x = data_original_i[:, 0].min()
        task_max_x = data_original_i[:, 0].max()
        sample(task, task_min_x, data_original_i, assumed_data_min, data_min)
        sample(task, task_max_x, data_original_i, assumed_data_max, data_max)

    scatter_min_diff.set_offsets(data_min)
    scatter_min_diff.set_sizes([100 for i in range(data_min.shape[0])])
    scatter_max_diff.set_offsets(data_max)
    scatter_max_diff.set_sizes([100 for i in range(data_max.shape[0])])


def regression_mean(i):

    data_min = numpy.zeros((N_POINTS * 5, 2))
    data_max = numpy.zeros((N_POINTS * 5, 2))

    for task in range(5):
        data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
        task_min_x = data_original_i[:, 0].min()
        task_max_x = data_original_i[:, 0].max()
        delta = (task_max_x - task_min_x) / 2
        data_min[task * N_POINTS:(task + 1) * N_POINTS, 0] = translate(
            assumed_data_min[:, task, 0],
            assumed_data_min[:, task, 0] + delta / 2,
            i, 50)
        data_min[task * N_POINTS:(task + 1) * N_POINTS, 1] = assumed_data_min[:, task, 1]
        data_max[task * N_POINTS:(task + 1) * N_POINTS, 0] = translate(
            assumed_data_max[:, task, 0],
            assumed_data_max[:, task, 0] - delta / 2,
            i, 50)
        data_max[task * N_POINTS:(task + 1) * N_POINTS, 1] = assumed_data_max[:, task, 1]
        
    scatter_min_diff.set_offsets(data_min)
    scatter_max_diff.set_offsets(data_max)


def results(i):

    data_min = numpy.zeros((N_POINTS * 5, 2))
    data_max = numpy.zeros((N_POINTS * 5, 2))

    for task in range(5):
        data_original_i = data_original[types.index('weights_init'), task, :N_POINTS]
        task_min_x = data_original_i[:, 0].min()
        task_max_x = data_original_i[:, 0].max()
        delta = (task_max_x - task_min_x) / 2
        data_min[task * N_POINTS:(task + 1) * N_POINTS, 0] = translate(
            assumed_data_min[:, task, 0] + delta / 2,
            assumed_data_min[:, task, 0] + delta,
            i, 50)
        data_min[task * N_POINTS:(task + 1) * N_POINTS, 1] = assumed_data_min[:, task, 1]
        data_max[task * N_POINTS:(task + 1) * N_POINTS, 0] = translate(
            assumed_data_max[:, task, 0] - delta / 2,
            assumed_data_max[:, task, 0] - delta,
            i, 50)
        data_max[task * N_POINTS:(task + 1) * N_POINTS, 1] = assumed_data_max[:, task, 1]
        
    scatter_min_diff.set_offsets(data_min)
    scatter_max_diff.set_offsets(data_max)


# 
# anim.save('sine_wave.gif', writer='imagemagick')

# im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
#                                    blit=True)


def main(frames):

    if frames:
        frames = range(*frames)
    else:
        frames = max_number_of_frames

    anim = FuncAnimation(fig, animate, # init_func=init,
                         frames=frames, interval=20, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Xavier Bouthillier'), bitrate=bitrate)
    fig.set_size_inches(width, height, True)
    anim.save('im.mp4', writer=writer, dpi=dpi)


if __name__ == '__main__':

    main([int(v) for v in sys.argv[1:]])
    #     # TODO: Use this argument
    # else:
    #     # Build the video
