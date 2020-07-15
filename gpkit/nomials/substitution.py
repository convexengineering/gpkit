"Scripts to parse and collate substitutions"
import warnings as pywarnings
import numpy as np
from ..small_scripts import splitsweep


def parse_subs(varkeys, substitutions, clean=False):
    "Seperates subs into the constants, sweeps, linkedsweeps actually present."
    constants, sweep, linkedsweep = {}, {}, {}
    if clean:
        for var in varkeys:
            if dict.__contains__(substitutions, var):
                sub = dict.__getitem__(substitutions, var)
                append_sub(sub, [var], constants, sweep, linkedsweep)
    else:
        varkeys.update_keymap()
        if hasattr(substitutions, "keymap"):
            for var in varkeys.keymap:
                if dict.__contains__(substitutions, var):
                    sub = dict.__getitem__(substitutions, var)
                    keys = varkeys.keymap[var]
                    append_sub(sub, keys, constants, sweep, linkedsweep)
        else:
            for var in substitutions:
                key = getattr(var, "key", var)
                if key in varkeys.keymap:
                    sub, keys = substitutions[var], varkeys.keymap[key]
                    append_sub(sub, keys, constants, sweep, linkedsweep)
    return constants, sweep, linkedsweep


def append_sub(sub, keys, constants, sweep, linkedsweep):
    "Appends sub to constants, sweep, or linkedsweep."
    sweepsub, sweepval = splitsweep(sub)
    if sweepsub:  # if the whole key is swept
        sub = sweepval
    for key in keys:
        if not key.shape or not getattr(sub, "shape", hasattr(sub, "__len__")):
            value = sub
        else:
            with pywarnings.catch_warnings():
                pywarnings.filterwarnings("error")
                try:
                    sub = np.array(sub) if not hasattr(sub, "shape") else sub
                except Warning:  # pragma: no cover  #TODO: coverage this
                    # ragged nested sequences, eg [[2]], [3, 4]], in py3.7+
                    sub = np.array(sub, dtype=object)
            if key.shape == sub.shape:
                value = sub[key.idx]
                sweepel, sweepval = splitsweep(value)
                if sweepel:  # if only an element is swept
                    value = sweepval
                    sweepsub = True
            elif sweepsub:
                try:
                    np.broadcast(sub, np.empty(key.shape))
                except ValueError:
                    raise ValueError("cannot sweep variable %s of shape %s"
                                     " with array of shape %s; array shape"
                                     " must either be %s or %s" %
                                     (key.veckey, key.shape, sub.shape,
                                      key.shape, ("N",)+key.shape))
                idx = (slice(None),)+key.descr["idx"]
                value = sub[idx]
            else:
                raise ValueError("cannot substitute array of shape %s for"
                                 " variable %s of shape %s." %
                                 (sub.shape, key.veckey, key.shape))
        if hasattr(value, "__call__") and not hasattr(value, "key"):
            linkedsweep[key] = value
        elif sweepsub:
            sweep[key] = value
        else:
            try:
                assert np.isnan(value)
            except (AssertionError, TypeError, ValueError):
                constants[key] = value
