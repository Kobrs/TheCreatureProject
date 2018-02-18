import re
import numpy as np

data_file = "log/runner_log_1000.txt"

cases = {}

with open(data_file, 'r') as f:
    for line in f:
        a = re.findall("(?<=architecture: )\w*", line)[0]
        m = re.findall("(?<=mean = )[0-9.-]*", line)[0]
        s = re.findall("(?<=stdev = )[0-9.-]*", line)[0]
        sc = float(re.findall("(?<=score = )[0-9.]*", line)[0])

        case_id = a + ' ' + m + ' ' + s
        try:
            cases[case_id].append(sc)
        except KeyError:
            cases[case_id] = [sc]



score_list = []
braitenbergs = []

for case_id, scores in cases.iteritems():
    score = sum(scores)/float(len(scores))
    score_list.append(score)
    if re.search("braitenberg", case_id):
        braitenbergs.append(score)

    print "Mean score for %s: %.2f"%(case_id, score)

best_id = cases.keys()[np.argmax(score_list)]
print "Best case: %s with score: %.2f"%(best_id, max(score_list))
 print "Best bratenberg:", max(braitenbergs)