"""Contains the ring road scenario class."""

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos, linspace
import numpy as np
import random

ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 230,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curves on the ring
    "resolution": 40
}


class TenaciousDScenario(Scenario):
    """Tenacious scenario.

    Requires from net_params:

    * **length** : length of the circle
    * **lanes** : number of lanes in the circle
    * **speed_limit** : max speed limit of the circle
    * **resolution** : number of nodes resolution

    See flow/scenarios/base_scenario.py for description of params.
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a loop scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        r = length / (2 * pi)

        nodes = [{
            "id": "bottom",
            "x": 0,
            "y": -r
        }, {
            "id": "right",
            "x": r+10,
            "y": 0
        }, {
            "id": "top",
            "x": 0,
            "y": r
        }, {
            "id": "left",
            "x": -r-10,
            "y": 0
        }, {
            "id": "left_upper_N",
            "x": -10,
            "y": r
        }, {
            "id": "left_lower_N",
            "x": -10,
            "y": -r
        }, {
            "id": "right_upper_N",
            "x": 10,
            "y": r
        }, {
            "id": "right_lower_N",
            "x": 10,
            "y": -r
        }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]
        resolution = net_params.additional_params["resolution"]
        r = length / (2 * pi)
        edgelen = length / 4.

        edges = [{
            "id":
                "dia",
            "type":
                "edgeType",
            "from":
                "bottom",
            "to":
                "top",
            "length":
                2*r
        },{
            "id":
                "right_upper_E",
            "type":
                "edgeType",
            "from":
                "top",
            "to":
                "right_upper_N",
            "length":
                10
        },{
            "id":
                "right_lower_E",
            "type":
                "edgeType",
            "from":
                "right_lower_N",
            "to":
                "bottom",
            "length":
                10
        },{
            "id":
                "left_upper_E",
            "type":
                "edgeType",
            "from":
                "top",
            "to":
                "left_upper_N",
            "length":
                10
        },{
            "id":
                "left_lower_E",
            "type":
                "edgeType",
            "from":
                "left_lower_N",
            "to":
                "bottom",
            "length":
                10
        },{
            "id":
                "right_upper",
            "type":
                "edgeType",
            "from":
                "right_upper_N",
            "to":
                "right",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t)+10, r * sin(t))
                    for t in linspace(pi / 2,0, resolution)
                ]
        }, {
            "id":
                "right_lower",
            "type":
                "edgeType",
            "from":
                "right",
            "to":
                "right_lower_N",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t)+10, r * sin(t))
                    for t in linspace(0,-pi / 2, resolution)
                ]
        }, {
            "id":
                "left_upper",
            "type":
                "edgeType",
            "from":
                "left_upper_N",
            "to":
                "left",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t)-10, r * sin(t))
                    for t in linspace(pi / 2, pi, resolution)
                ]
        }, {
            "id":
                "left_lower",
            "type":
                "edgeType",
            "from":
                "left",
            "to":
                "left_lower_N",
            "length":
                edgelen,
            "shape":
                [
                    (r * cos(t)-10, r * sin(t))
                    for t in linspace(pi, 3 * pi / 2, resolution)
                ]
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{
            "id": "edgeType",
            "numLanes": lanes,
            "speed": speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "right_upper": ["right_upper","right_lower","right_lower_E","dia","right_upper_E"],
            "right_lower": ["right_lower","right_lower_E","dia","right_upper_E","right_upper"],
            "left_upper": ["left_upper","left_lower","left_lower_E","dia","left_upper_E"],
            "left_lower": ["left_lower","left_lower_E","dia","left_upper_E","left_upper"],
            "right_upper_E": ["right_upper_E","right_upper","right_lower","right_lower_E","dia"],
            "right_lower_E": ["right_lower_E","dia","right_upper_E","right_upper","right_lower",],
            "left_upper_E": ["left_upper_E","left_upper","left_lower","left_lower_E","dia"],
            "left_lower_E": ["left_lower_E","dia","left_upper_E","left_upper","left_lower"]
        }

        return rts

    #def specify_edge_starts(self):
    #    """See parent class."""
    #    length = 230
    #    edgelen = self.length / 4
    #    s = 8
    #    r = length / (2 * pi)
    #    l = 8
    #    edgestarts = [("dia", 0),
    #                  ("right_upper_E", 2*r+s),             ("left_upper_E",  2*r+l),
    #                  ("right_upper",   2*r+s+10),          ("left_upper",    2*r+l+10),
    #                  ("right_lower",   2*r+s+10+edgelen),  ("left_lower",    2*r+l+10+edgelen),
    #                  ("right_lower_E", 2*r+s+10+edgelen*2), ("left_lower_E",  2*r+l+10+edgelen*2)]
    #
    #    return edgestarts

    def specify_edge_starts(self):
        """See parent class."""
        length = 230
        edgelen = self.length / 4
        s = 6
        r = length / (2 * pi)
        l = 6
        edgestarts = [("dia", edgelen+10+s),
                      ("right_upper_E", edgelen+10+s+2*r+s),   ("left_upper_E",  edgelen+10+s+2*r+s+0.01),
                      ("right_upper",   edgelen+10+s+2*r+s+10),("left_upper",    edgelen+10+s+2*r+s+10+0.01),
                      ("right_lower",   0),                    ("left_lower",    0.01),
                      ("right_lower_E", edgelen),              ("left_lower_E",  edgelen+0.01)]

        return edgestarts

    #def specify_intersection_edge_starts(self):
    #    length = 230
    #    edgelen = self.length / 4
    #    s = 8
    #    r = length / (2 * pi)
    #    l = 8
    #    intersection_edgestarts = \
    #        [(":bottom_1",2*r+s+10+edgelen*2+10),
    #        (":bottom_0",2*r+s+10+edgelen*2+10.01),
    #        (":top_1",2*r),
    #        (":top_0",2*r+0.01)]
    #    return intersection_edgestarts

    def specify_intersection_edge_starts(self):
        length = 230
        edgelen = self.length / 4
        s = 6
        r = length / (2 * pi)
        l = 6
        intersection_edgestarts = \
            [(":bottom_1",10+s+edgelen),
            (":bottom_0",10+s+edgelen+0.01),
            (":top_1",10+s+edgelen+2*r),
            (":top_0",10+s+edgelen+2*r+0.01)]
        return intersection_edgestarts

    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        startpositions = []
        startlanes = []
        length = 230
        edgelen = self.length / 4

        nr_left  = num_vehicles // 2 + num_vehicles % 2
        nr_right = num_vehicles // 2
        pos_left  = np.linspace(5,edgelen-5,num=nr_left)
        pos_right = np.linspace(5,edgelen-5,num=nr_right)

        for pos in np.nditer(pos_left):
            startpositions.append(("left_lower",pos+random.uniform(-15,15)))
            startlanes.append(0)
        for pos in np.nditer(pos_right):
            startpositions.append(("right_lower",pos+random.uniform(-15,15)))
            startlanes.append(0)
        print(startpositions)
        return startpositions, startlanes



