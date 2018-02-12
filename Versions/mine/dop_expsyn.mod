NEURON {
    POINT_PROCESS dop_ExpSyn
    RANGE tau, e, i, dopdr, dop
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau = 0.21 (ms) <1e-9,1e9>
    e = 0   (mV)
    dopdr = 0.01
    dop = 0
}

ASSIGNED {
    v (mV)
    i (nA)
}

STATE {
    g (uS)
}

INITIAL {
    g=0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    : For now let's use linear function
    dop = dop - dopdr 
    if (dop < 0) {
        dop = 0
    }
    i = g*(v - e)
}

DERIVATIVE state {
    g' = -g/tau
}

NET_RECEIVE(weight (uS)) {
    g = g + (weight*(1+dop))
}
