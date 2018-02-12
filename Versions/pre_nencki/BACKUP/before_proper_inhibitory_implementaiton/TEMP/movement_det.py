def detector(spikes, alpha=0.6, beta=4., th=12.):
    mov_list = []

    val = 0
    for i in xrange(len(spikes)):
        sp = spikes[i]
        prev_sp = spikes[i-1] if i>0 else 0

        dt = sp - prev_sp
        val += beta
        # Decay
        val -= dt*alpha

        if val < 0:
            val = 0

        print "spike at time:", sp, "current detector value:", val,
        if val >= th:
            mov_list.append(i)
            print "movement!"
        else:
            print ""

    return mov_list

spikes = [10, 20, 25, 27, 30, 33, 36, 39, 50, 52, 55, 56, 59, 100, 120]
detector(spikes, alpha=0.6, beta=4., th=12)