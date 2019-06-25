from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

if __name__ == '__main__':
    # Defining the model structure. We can define the network by just passing a list of edges.
    model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

    # Defining individual CPDs.
    cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
    cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])
    cpd_g = TabularCPD(variable='G', variable_card=3,
                       values=[[0.3, 0.05, 0.9,  0.5],
                               [0.4, 0.25, 0.08, 0.3],
                               [0.3, 0.7,  0.02, 0.2]],
                       evidence=['I', 'D'],
                       evidence_card=[2, 2])

    cpd_l = TabularCPD(variable='L', variable_card=2,
                       values=[[0.1, 0.4, 0.99],
                               [0.9, 0.6, 0.01]],
                       evidence=['G'],
                       evidence_card=[3])

    cpd_s = TabularCPD(variable='S', variable_card=2,
                       values=[[0.95, 0.2],
                               [0.05, 0.8]],
                       evidence=['I'],
                       evidence_card=[2])

    # Associating the CPDs with the network
    model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
    # print model.check_model()
    # print model.get_cpds()
    # print(model.get_cpds('G'))
    # print model.get_cardinality('G')