import numpy
import scipy.stats


def pab(pa, pb):
    return (pa > pb).mean()


def normal_ci(pa, pb, sample_size=None, alpha=0.05):
    if sample_size is None:
        sample_size = pa.shape[0]
    p_a_b = pab(pa, pb)
    return scipy.stats.norm.isf(alpha / 2) * numpy.sqrt(
        p_a_b * (1 - p_a_b) / sample_size
    )


def percentile_bootstrap(pa, pb, alpha=0.05, bootstraps=None):
    if len(pa.shape) < 2:
        pa = pa.reshape((-1, 1))
        pb = pb.reshape((-1, 1))

    sample_size = pa.shape[0]
    simuls = pa.shape[1]

    if bootstraps is None:
        bootstraps = sample_size

    stats = numpy.zeros((bootstraps, simuls))
    for i in range(bootstraps):
        idx = numpy.random.randint(0, sample_size, size=sample_size)
        stats[i] = (pa[idx, :] > pb[idx, :]).mean(0)

    stats = numpy.sort(stats, axis=0)
    lower = numpy.percentile(stats, alpha / 2 * 100, axis=0)
    upper = numpy.percentile(stats, (1 - alpha / 2) * 100, axis=0)

    return lower, upper
