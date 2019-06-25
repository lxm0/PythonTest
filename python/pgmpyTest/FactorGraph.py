import numpy as np
from networkx import Graph

from pgmpy.factors.discrete import DiscreteFactor

from pgmpy.inference import BeliefPropagation

from pgmpy.models import FactorGraph


if __name__ == '__main__':
  G = FactorGraph()
  G.add_nodes_from(['x2', 'x3', 'x4'])
  phi3 = DiscreteFactor(['x2', 'x3'], [2, 2],values=np.array([0.95,0.95,0.05,0.95]))
  phi4 = DiscreteFactor(['x2', 'x3'], [2, 2],values=np.array([0.8,0.2,0.8,0.8]))
  phi5 = DiscreteFactor(['x2', 'x4'], [2, 2], values=np.array([0.7,0.3,0.7,0.7]))
  phi7 = DiscreteFactor(['x3'], [2], values=np.array([0.0,1.0]))
  phi8 = DiscreteFactor(['x4'], [2], values=np.array([0.2,0.8]))
  G.add_factors(phi3, phi4,phi5,phi7,phi8)
  G.add_nodes_from([phi3, phi4,phi5,phi7,phi8])
  G.add_edges_from([('x2', phi3),('x2', phi4),('x2', phi5) ,('x3', phi3),('x3', phi4), ('x3', phi7),('x4', phi5),('x4', phi8)])
  print(G.number_of_edges())
  print(G.size())
  # print G.check_model()
  # print G.get_partition_function()
  # bp = BeliefPropagation(G)





