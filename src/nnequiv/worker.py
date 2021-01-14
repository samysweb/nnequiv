import time

from nnenum.timerutil import Timers

from nnenum.worker import Worker

class EquivWorker(Worker):
    def __init__(self, shared, priv):
        super(EquivWorker,self).__init__(shared,priv)
    
    def check_is_finished(self):
        network1 = self.shared.network1
        network2 = self.shared.network2
        network = [network1, network2]
        return self.priv.ss.is_finished(network)
    
    def advance_star(self):
        '''advance current star (self.priv.ss)

        A precondition to this is that ss is already at the next split point.

        The logic for this is:

        1. do split, creating new_star
        2. propagate up to next split with ss
        3. propagate up to next split with new_star
        4. save new_star to remaining work
        '''

        Timers.tic('advance')

        ss = self.priv.ss
        network = [self.shared.network1,self.shared.network2]
        spec = self.shared.spec

        if not self.check_is_finished():
            new_star = ss.do_first_relu_split(network, spec, self.priv.start_time)

            ss.propagate_up_to_split(network, self.priv.start_time)
            
            if new_star: # new_star can be null if it wasn't really a split (copy prefilter)
                new_star.propagate_up_to_split(network, self.priv.start_time)
                # note: new_star may be done... but for expected branching order we still add it
                self.priv.stars_in_progress += 1
                self.priv.work_list.append(new_star)

        Timers.toc('advance')

    def finished_star(self):
        'finished with a concrete star state'

        Timers.tic('finished_star')

        cur_star_set = self.priv.ss
        # print("Finished star set: {}",str(cur_star_set.output_stars))

        self.priv.num_lps_enum += cur_star_set.star.num_lps
        self.priv.finished_stars += 1
        self.priv.num_lps += cur_star_set.star.num_lps

        #self.add_branch_str(f"concrete {'UNSAFE' if violation_star is not None else 'safe'}")

        self.priv.ss = None
        # local stats that get updates in update_shared_variables
        self.priv.update_stars += 1
        self.priv.update_work_frac += cur_star_set.work_frac
        self.priv.update_stars_in_progress -= 1

        if not self.priv.work_list:
            # urgently update shared variables to try to get more work
            self.priv.shared_update_urgent = True
            self.priv.fulfillment_requested_time = time.perf_counter()
        
        # if Settings.PRINT_BRANCH_TUPLES:
        #     print(self.priv.branch_tuples_list[-1])

        # if violation_star is not None:
        #     self.found_unsafe(concrete_io_tuple)

        Timers.toc('finished_star')
        self.update_shared_variables()
