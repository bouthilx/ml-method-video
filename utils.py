import numpy


def sigmoid(t):
    return 1 / (1 + numpy.exp(-t))


def linear(a, b, step, steps):
    return a + step / steps * (b - a)


def translate(a, b, step, steps, saturation=10):
    return a + (sigmoid(step / steps * saturation) - 0.5) * 2 * (b - a)
