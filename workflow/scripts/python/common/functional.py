from typing import TypeVar, Callable, Iterable

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


def lookup_maybe(k: X, xs: dict[X, Y]) -> Y | None:
    return xs[k] if k in xs else None


def from_maybe(x: X, y: X | None) -> X:
    return x if y is None else y


def fmap_maybe(f: Callable[[X], Y], x: X | None) -> Y | None:
    return f(x) if x is not None else None


def fmap_maybe_def(c: Y, f: Callable[[X], Y], x: X | None) -> Y:
    return f(x) if x is not None else c


def lookup_map(f: Callable[[Y], Z | None], k: X, xs: dict[X, Y]) -> Z | None:
    return fmap_maybe(f, lookup_maybe(k, xs))


def float_maybe(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def span(f: Callable[[X], bool], xs: list[X]) -> tuple[list[X], list[X]]:
    if len(xs) == 0:
        return ([], [])
    n = 0
    while f(xs[n]):
        n = n + 1
    return (xs[:n], xs[n:])


def partition(f: Callable[[X], bool], xs: Iterable[X]) -> tuple[list[X], list[X]]:
    ys = []
    zs = []
    for x in xs:
        if f(x):
            ys.append(x)
        else:
            zs.append(x)
    return (ys, zs)


def unzip2(xs: list[tuple[X, Y]]) -> tuple[list[X], list[Y]]:
    return [x[0] for x in xs], [x[1] for x in xs]
