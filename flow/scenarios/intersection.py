"""Contains the intersection scenario class."""

import numpy as np
import random
from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

ADDITIONAL_NET_PARAMS = {
    # length of each edge
    "edge_length": 100,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30
}


class IntersectionTWScenario(Scenario):
    """intersection scenario class."""
	
    def __init___(self,
                  name,
                  vehicles,
                  net_params,
                  initial_config=InitialConfig(),
                  traffic_lights=TrafficLightParams()):
	
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))
    
        self.edge_length = net_params.additional_params["edge_length"]
        self.lanes = net_params.additional_params["lanes"]
        self.inner_space_len = 0.28
    
        # instantiate "length" in net params
        net_params.additional_params["length"] = 4 * self.edge_length
        
        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        l = net_params.additional_params["edge_length"]

        nodes = [{
            "id": "intersection",
            "x": repr(0),
            "y": repr(0),
            "type": "priority"
        }, {
            "id": "top",
            "x": repr(0),
            "y": repr(l),
            "type": "priority"
        }, {
            "id": "bottom",
            "x": repr(0),
            "y": repr(-l),
            "type": "priority"
        }, {
            "id": "right",
            "x": repr(4*l),
            "y": repr(0),
            "type": "priority"
        }]

        return nodes
        
    def specify_edges(self, net_params):
        """See parent class."""
        l = net_params.additional_params["edge_length"]

        edges = [{
            "id": "top_intersection",
            "type": "edgeType",
            "from": "top",
            "to": "intersection",
            "length": repr(l),
            "priority": "78"
        }, {
            "id": "bottom_intersection",
            "type": "edgeType",
            "from": "bottom",
            "to": "intersection",
            "length": repr(l),
            "priority": "79"
        }, {
            "id": "intersection_right",
            "type": "edgeType",
            "from": "intersection",
            "to": "right",
            "length": repr(4*l),
            "priority": "46"
        }]
        return edges
        
    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": repr(lanes),
            "speed": repr(speed_limit)
        }]
        return types
    
    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "intersection_right": ["intersection_right"],
            "bottom_intersection": ["bottom_intersection", "intersection_right"],
            "top_intersection": ["top_intersection", "intersection_right" ]
        }
        return rts
        
    def specify_edge_starts(self):
        """See base class."""
        l = 40
        jl = 2.9+3.3
        
        edgestarts = \
            [("top_intersection",0),
             ("bottom_intersection",0),
             ("intersection_right",l+jl)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        intersection_edgestarts = \
            [(":intersection_1",40),
            (":intersection_0",40.01)]
    
        return intersection_edgestarts

    def specify_internal_edge_starts(self):
        internal_edgestarts = \
          [(":top_intersection",0),
           (":bottom_intersection",0),
           (":intersection_right",0)]

        return internal_edgestarts
     
    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        perturbation_1 = initial_config.perturbation * random.random() 
        perturbation_2 = initial_config.perturbation * random.random()
        print(perturbation_1)
        print(perturbation_2)
        startpositions = []
        startlanes = []
        startpositions.append(("top_intersection",0 + perturbation_1))
        startlanes.append(0)
        startpositions.append(("bottom_intersection",0 + perturbation_2))
        startlanes.append(0)
        return startpositions, startlanes

    def specify_connections(self, net_params):
        connections = []
        connections.append({ "from": "top_intersection",
                             "to": "intersection_right",
                             "pass": "false"})
        connections.append({ "from": "bottom_intersection",
                             "to": "intersection_right",
                             "pass": "true"})
        return connections

    
