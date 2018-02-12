NEURON {
    SUFFIX iafI
    RANGE tau, refrac, i, m_th, m_rest
}


UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
}


PARAMETER {
	tau = 10 (ms)
	refrac = 5 (ms)
  i = 0 (nA)
  v_th = 15 (mV)
  m_rest = 0 (mV)
}

: Not sure about how ponter will work from here 
ASSIGNED {
	t0(ms)
	refractory
}


STATE {
    m (mV)
}


INITIAL {
  : set it to resting potential
  t0 = t
  m = m_rest
  refractory = 0 : 0-integrates input, 1-refractory
}


BREAKPOINT {
  SOLVE state METHOD cnexp
  m = m*1
  if (m > v_th) {
    m = m_rest  : reset membrane voltage to resting potential
  }
}


DERIVATIVE state {
  m' = -(m-i) / tau
}

