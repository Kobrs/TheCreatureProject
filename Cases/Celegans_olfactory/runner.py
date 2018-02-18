"""
This code will spawn many instances of main.py with different parameters passed
using commandline arguments and read out the last line of stdout, which
should be creature score.
Its main purpose is to perform long runs searching for optimal parameters
or comparing setup performance.
"""
import subprocess
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', type=str)
ag = parser.parse_args()


max_num_processes = 6
log_file = "log/runner_log.txt"

score_time = 20000
num_repeats = 12

stdp = False

lsources = 1
# Set searched parameters ranges and amount of values tested
# (min_val, max_val, num_of_parameter_values)
# noise_mean_conf = [-0.3, 0.1, 5]
# noise_stdev_conf = [0, 1, 5]
# architectures = ["prebuilt", "braitenberg"]
noise_mean_conf = [-0.1, -0.1, 1]
noise_stdev_conf = [0.75, 0.75, 1]
architectures = ["prebuilt"]

# Keep track of all spawned processes, so we can recover scores
processes = []
running_porcesses = []  # used to maintain max_num_processes at once
scores = []  # it'll contain tuples of [mean, stdev, score]



def spawn_instance(noise_mean, noise_stdev, architecture, display=False, frozen=False):
    """
    Spawns instance of main.py with global and given parameters and returns
    process, so stdout can be retrived
    """
    cmd = "python main.py --silent --score_time %d --lsources %d\
          --noise_mean %d --noise_stdev %d --architecture %s"%(
            score_time, lsources, noise_mean, noise_stdev, architecture)
    if not stdp: cmd += ' --no_STDP'
    if not display: cmd += ' --not_display'
    if frozen: cmd += ' --frozen'

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    return process


def get_score(process):
    """
    Returns last line of stdout, which should be score, as an integer. Note
    that this also synchronises processes, so you should call this only
    after spawning all instances.
    """
    # Just in case something brakes
    try:
        return int(process.stdout.readlines()[-1])
    except:
        return None



def grid_search():
    with open(log_file, 'w') as f:
        for m in np.linspace(*noise_mean_conf):
            for s in np.linspace(*noise_stdev_conf):
                for a in architectures:
                    for repeat_i in xrange(num_repeats):
                        print "Testing m=%f s=%f, a=%s, repeat=%d"%(m, s, a, repeat_i)
                        while True:
                            if len(running_porcesses) < max_num_processes:
                                process = spawn_instance(m, s, a)
                                processes.append(process)
                                running_porcesses.append((m, s, a, process))
                                break
                            else:
                                # Check if any process finished
                                for i, msap in enumerate(running_porcesses):
                                    m_, s_, a_,  p_ = msap
                                    if p_.poll() is not None:
                                        # Score finished
                                        score = get_score(p_)
                                        scores.append((m_, s_, a_, score))
                                        f.write(("Scoring: architecture: %s " +
                                                 "mean = %f, stdev = %f, score = %f\n")%(
                                                  a_,  m_, s_, score))
                                        print (("Scoring: architecture: %s " +
                                                 "mean = %f, stdev = %f, score = %f\n")%(
                                                  a_,  m_, s_, score))
                                        # Write changes to file so it's possible to keep track
                                        f.flush()

                                        del running_porcesses[i]



def many_instances(n, noise_mean, noise_stdev, architecture):
    for i in xrange(n):
        spawn_instance(noise_mean, noise_stdev, architecture, display=True, frozen=True)





if ag.mode == "grid_search":
    grid_search()

    print "running_processes:", running_porcesses
    # Get output of ALL processes
    with open(log_file, 'a') as f:
        for msap in running_porcesses:
            m, s, a, p = msap
            score = get_score(p)
            scores.append((m, s, a, score))
            f.write("Scoring: architecture: %s mean = %f, stdev = %f, score = %f\n"%(
                    a, m, s, score))


    print scores

elif ag.mode == "many_instances":
    lsources = 3
    score_time = 1e9
    many_instances(n=max_num_processes, noise_mean=-0.1, noise_stdev=1, architecture="all2all")