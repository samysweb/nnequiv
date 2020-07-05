'''
Measurement script for ACAS Xu networks
'''

import sys
import time
from termcolor import cprint

from nnenum.settings import Settings
from acasxu_single import verify_acasxu

def main():
    'main entry point'

    Settings.TIMING_STATS = False
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = False

    full_filename = 'full_acasxu.dat'
    hard_filename = 'hard_acasxu.dat'
    accumulated_filename = 'accumulated_full.dat'
    hard_accumulated_filename = 'accumulated_hard.dat'

    if len(sys.argv) > 1:
        Settings.TIMEOUT = 60 * float(sys.argv[1])
        print(f"Running measurements with timeout = {Settings.TIMEOUT} secs")

    finished_times = [] # for results that finished
    finished_hard = [] # for results that finished

    start = time.time()

    instances = []

    for spec in range(1, 5):
        for a_prev in range(1, 6):
            for tau in range(1, 10):
                instances.append([str(a_prev), str(tau), str(spec)])

    instances.append(["1", "1", "5"])
    instances.append(["1", "1", "6"])
    instances.append(["1", "9", "7"])
    instances.append(["2", "9", "8"])
    instances.append(["3", "3", "9"])
    instances.append(["4", "5", "10"])

    acasxu_hard = [["4", "6", "1"],
                   ["4", "8", "1"],
                   ["3", "3", "2"],
                   ["4", "2", "2"],
                   ["4", "9", "2"],
                   ["3", "6", "3"],
                   ["5", "1", "3"],
                   ["1", "9", "7"],
                   ["3", "3", "9"]]



    with open(hard_filename, "w") as h:
        with open(full_filename, "w") as f:
            for instance in instances:
                a_prev, tau, spec = instance
                net_pair = (int(a_prev), int(tau))

                res_str = "none"
                secs = -1

                cprint(f"\nRunning net {a_prev}-{tau} with spec {spec}", "grey", "on_green")

                if spec != "7":
                    res_str, secs = verify_acasxu(net_pair, spec)
                else:
                    # spec 7 is nondeterministic due to work sharing among processes... use best of several runs
                    pretimeout = Settings.TIMEOUT
                    Settings.TIMEOUT = 3
                    best_time = np.inf
                    best_res = None

                    for i in range(10):
                        print(f"Doing property 7 quick trial #{i}")

                        res_str, secs = verify_acasxu(net_pair, spec)

                        if res_str != "timeout":
                            best_time = min(best_time, secs)
                            best_res = res_str

                    Settings.TIMEOUT = pretimeout

                    if best_res is None:
                        # none succeeded... use original timeout
                        res_str, secs = verify_acasxu(net_pair, spec)
                    else:
                        res_str = best_res
                        secs = best_time

                s = f"{a_prev}_{tau}\t{spec}\t{res_str}\t{secs}"
                f.write(s + "\n")
                f.flush()
                print(s)

                if instance in acasxu_hard:
                    h.write(s + "\n")
                    h.flush()

                if res_str not in ["timeout", "error"]:
                    finished_times.append(secs)

                    if instance in acasxu_hard:
                        finished_hard.append(secs)


    # save accumulated results
    finished_times.sort()
    finished_hard.sort()

    with open(accumulated_filename, "w") as f:
        for i, secs in enumerate(finished_times):
            f.write(f"{secs}\t{i+1}\n")

    with open(hard_accumulated_filename, "w") as f:
        for i, secs in enumerate(finished_hard):
            f.write(f"{secs}\t{i+1}\n")

    mins = (time.time() - start) / 60.0

    print(f"Completed all measurements in {round(mins, 2)} minutes")

if __name__ == '__main__':
    main()