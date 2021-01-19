import copy

import numpy as np

from nnenum.lp_star import LpStar
from nnenum.lp_star_state import LpStarState
from nnenum.network import ReluLayer
from nnenum.lpinstance import LpInstance

StarSetId = 1

class EquivStarState(LpStarState):
    def __init__(self, network_count, from_star_state=None):
        global StarSetId
        self._fully_initialized=False
        self.id = StarSetId
        self.from_id=0
        StarSetId+=1
        if from_star_state is not None:
            self.star = from_star_state.star
            self.prefilter = copy.deepcopy(from_star_state.prefilter)
            self.cur_layer = from_star_state.cur_layer
            self.work_frac = from_star_state.work_frac
            self.should_try_overapprox = from_star_state.should_try_overapprox
            self.safe_spec_list = from_star_state.safe_spec_list
            self.branch_tuples = from_star_state.branch_tuples
            self.distance_to_unsafe = from_star_state.distance_to_unsafe
            self.cur_network = 0
            self.output_stars = []
            for x in range(0,network_count):
                self.output_stars.append(None)
            self.network_count = network_count
            self.initial_star=None
            self._fully_initialized=True
            self.freeze_attrs()
        else:
            super(EquivStarState,self).__init__()
            self.cur_network = 0
            self.output_stars = []
            for x in range(0,network_count):
                self.output_stars.append(None)
            self.initial_star=None
            self.network_count=network_count
            self._fully_initialized=True
            self.freeze_attrs()
    
    def freeze_attrs(self):
        if self._fully_initialized:
            self._frozen = True
    
    def from_init_star(self, star):
        super(EquivStarState,self).from_init_star(star)
        self.initial_star = star.copy()
    
    def propagate_up_to_split(self, networks, start_time):
        assert len(networks) > 1 and len(networks)>self.cur_network
        depth = len(self.branch_tuples)
        
        while not self.is_finished(networks):
            network = networks[self.cur_network]
            layer = network.layers[self.cur_layer]
            
            if isinstance(layer, ReluLayer):
                if self.prefilter.output_bounds is None:
                    # start of a relu layer
                    self.prefilter.init_relu_layer(self.star, layer, start_time, depth)

                if self.prefilter.output_bounds.branching_neurons.size > 0:
                    break

                self.next_layer()
            else:
                # non-relu layer
                self.apply_linear_layer(network)
                
                self.next_layer()
        

    
    def split_enumerate(self, i, network, spec, start_time):
        rv = super(EquivStarState,self).split_enumerate(i, network, spec, start_time)
        if rv is None:
            return None
        rv = EquivStarState(self.network_count,
            from_star_state=rv
        )
        rv.initial_star=self.initial_star
        rv.from_id = self.id
        rv.cur_network = self.cur_network
        
        # TODO(steuber): Recheck this procedure
        for x in range(0, self.cur_network):
            rv.output_stars[x]=self.output_stars[x]
        # rv.star.check_input_box_bounds_slow()
        return rv
    
    def do_first_relu_split(self, networks, spec, start_time):
        assert len(networks) > 1 and len(networks)>self.cur_network
        
        return super(EquivStarState,self).do_first_relu_split(networks[self.cur_network], spec, start_time)

    def is_finished(self, networks):
        'is the current star finished?'
        assert len(networks) > 1 


        if self.cur_network<self.network_count and self.cur_layer >= len(networks[self.cur_network].layers):
            self.cur_layer=0
            self.output_stars[self.network_count-1]=self.star
            new_in_star = LpStar(
               self.initial_star.a_mat,
               self.initial_star.bias,
               box_bounds=None,
               lpi=LpInstance(self.star.lpi)
            )
            new_in_star.input_bounds_witnesses = copy.deepcopy(self.star.input_bounds_witnesses)
            new_in_star.input_bounds_witnesses = self.star.input_bounds_witnesses
            # new_in_star = self.initial_star.copy()
            # new_in_star.lpi = LpInstance(self.star.lpi)
            self.cur_network += 1
            self.from_init_star(new_in_star)

        if self.network_count<=self.cur_network:
            return True
        else:
            return False
        

    