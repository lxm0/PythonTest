from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
if __name__ == '__main__':
    student_model = BayesianModel([('D', 'G'), ('I', 'G')])

    difficulty_cpd = TabularCPD(variable='D', variable_card=2, values=[[0.6, 0.4]])
    intel_cpd = TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.3]])
    grade_cpd = TabularCPD(variable='G',variable_card=3, values=[[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]], evidence=['I', 'D'], evidence_card=[2, 2])

    student_model.add_cpds(grade_cpd, difficulty_cpd, intel_cpd)

    print (student_model.nodes())
    print (student_model.get_cpds('D'))
    print (student_model.get_cpds('I'))
    print (student_model.get_cpds('G'))

    belief_propagation = BeliefPropagation(student_model)
    res = belief_propagation.query(variables=["G"])
    print (res['G'])