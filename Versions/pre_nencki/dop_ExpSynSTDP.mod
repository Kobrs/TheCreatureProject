NEURON {
    POINT_PROCESS dop_ExpSynSTDP
    RANGE tau, e, i, d, p, dtau, ptau, dop, dopdr
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau = 0.1 (ms) <1e-9,1e9>
    e = 0   (mV)
    d = 0 <0,1>: depression factor (multiplicative to prevent < 0)
    p = 0 : potentiation factor (additive, non-saturating)
    dtau = 34 (ms) : depression effectiveness time constant
    ptau = 17 (ms) : Bi & Poo (1998, 2001)
    dopdr = 0.01
    dop = 0  : dopamine mechanism disabled
}

ASSIGNED {
    v (mV)
    i (nA)
    tpost (ms)
}

STATE {
    g (uS)
}

INITIAL {
    g=0
    tpost = -1e9
    net_send(0, 1)
}

BREAKPOINT {
    SOLVE state METHOD cnexp

    : Linear dopamine drop
    dop = dop - dopdr 
    if (dop < 0) {
        dop = 0
    }

    i = g*(v - e)
}

DERIVATIVE state {
    g' = -g/tau
}

NET_RECEIVE(w (uS), A, tpre (ms)) {
    INITIAL { A = 0  tpre = -1e9 }
    if (flag == 0) { : presynaptic spike  (after last post so depress)
: printf("entry flag=%g t=%g w=%g A=%g tpre=%g tpost=%g\n", flag, t, w, A, tpre, tpost)
        g = g + w*(1 + A)*(1 + dop) : both stdp and dopamine modifies g
        tpre = t
        A = A * (1 - d*exp((tpost - t)/dtau))
    }else if (flag == 2) { : postsynaptic spike
: printf("entry flag=%g t=%g tpost=%g\n", flag, t, tpost)
        tpost = t
        FOR_NETCONS(w1, A1, tp) { : also can hide NET_RECEIVE args
: printf("entry FOR_NETCONS w1=%g A1=%g tp=%g\n", w1, A1, tp)
            A1 = A1 + p*exp((tp - t)/ptau)
        }
    } else { : flag == 1 from INITIAL block
: printf("entry flag=%g t=%g\n", flag, t)
        WATCH (v > -20) 2
    }
}
