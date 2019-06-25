import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import TabularCPD
if __name__ == '__main__':
    cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
                                [0.1,0.1,0.1,0.1,0.1,0.1],
                                [0.8,0.8,0.8,0.8,0.8,0.8]],
                     evidence=['diff', 'intel'], evidence_card=[2,3])

    # print cpd.variable
    # print cpd.variable_card
    # print cpd.get_evidence()
    # print cpd.values
    # print cpd.variables
    # print cpd
    # print cpd.reorder_parents(['intel', 'diff'])
    # print cpd
    # cpd_table = TabularCPD('grade', 2,
    #                                             [[0.7, 0.2, 0.6, 0.2],[0.4, 0.4, 0.4, 0.8]],
    #                                         ['intel', 'diff'], [2, 2])
    # print cpd_table
    # cpd_table.normalize()
    # print cpd_table.get_values()
    # cpd_table.reduce([('diff', 0)])
    # print cpd_table
    # print cpd_table.get_values()
    # G = FactorGraph()
    # G.add_node('a')
    # G.add_nodes_from(['a', 'b'])
    # phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
    # G.add_factors(phi1)
    # G.add_nodes_from([phi1])
    # G.add_edge('a', phi1)
    # G.add_edges_from([('a', phi1), ('b', phi1)])
    # print G.get_cardinality('a')
    import numpy as np
    # from pgmpy.factors.discrete import DiscreteFactor
    # # phi = DiscreteFactor(['diff', 'intel'], [2, 2], np.ones(4))
    # # print phi
    # # print phi.assignment([1, 3])
    # # phi = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 3], np.arange(18))
    # # phi_copy = phi.copy()
    # # print phi_copy.variables
    # # print phi
    # # print phi.values
    # phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    # phi2 = DiscreteFactor(['x3', 'x1'], [2, 2], range(1, 5))
    # print phi1
    # print phi2
    # phi1.divide(phi2)
    # print phi1
    # print phi1.variables
    # print phi1.cardinality
    # print phi1.values
    from pgmpy.models import MarkovModel
    from pgmpy.factors.discrete import DiscreteFactor
    from pgmpy.inference import Mplp, mplp
    import numpy as np
    student = MarkovModel()
    student.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F')])
    factor_a = DiscreteFactor(['A'], cardinality=[2], values=np.array([0.54577, 1-0.54577]))
    factor_b = DiscreteFactor(['B'], cardinality=[2], values=np.array([0.93894,1- 0.93894]))
    factor_c = DiscreteFactor(['C'], cardinality=[2], values=np.array([0.89205,1-0.89205]))
    factor_d = DiscreteFactor(['D'], cardinality=[2], values=np.array([0.56292, 1-0.56292]))
    factor_e = DiscreteFactor(['E'], cardinality=[2], values=np.array([0.47117, 1-0.47117]))
    factor_f = DiscreteFactor(['F'], cardinality=[2], values=np.array([0.5093, 1-0.5093]))
    # print factor_f
    # factor_a_b = DiscreteFactor(['A', 'B'], cardinality=[2, 2],
    #                                                     values=np.array([1.3207, 0.75717, 0.75717, 1.3207]))
    # factor_a_b.normalize()
    # print factor_a_b
    # factor_b_c = DiscreteFactor(['B', 'C'], cardinality=[2, 2],
    #                                                         values=np.array([0.00024189, 4134.2, 4134.2, 0.0002418]))
    # factor_b_c.normalize()
    # factor_c_d = DiscreteFactor(['C', 'D'], cardinality=[2, 2],
    #                                                          values=np.array([0.0043227, 231.34, 231.34, 0.0043227]))
    # factor_c_d.normalize()
    # factor_d_e = DiscreteFactor(['E', 'F'], cardinality=[2, 2],
    #                                                       values=np.array([31.228, 0.032023, 0.032023, 31.228]))
    # factor_d_e.normalize()
    # student.add_factors(factor_a, factor_b, factor_c, factor_d, factor_e, factor_f,factor_a_b, factor_b_c, factor_c_d, factor_d_e)
    # print student.factors
    # mplp = Mplp(student)
    # result = mplp.map_query()
    # print mplp.variables
    # print result

