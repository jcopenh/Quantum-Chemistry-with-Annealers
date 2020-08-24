import numpy as np
import pandas as pd
import scipy as sci
import symengine as se
import dimod, itertools
from openfermion.transforms import get_sparse_operator
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits, commutator

from helper_functions import *



#Returns coherent state evaluated at given bloch angles
def get_coherent_state(bloch_angles, n):
    coherent_state = 1
    for i in range(n):
        try:
            phi = bloch_angles[se.Symbol('phi'+str(i))]
        except KeyError:
            phi = 0
        the = bloch_angles[se.Symbol('the'+str(i))]

        psi = np.cos(the/2)*np.array([1,0]) + np.exp(1j*phi)*np.sin(the/2)*np.array([0,1])
        
        coherent_state = np.kron(coherent_state, psi)
    coherent_state = np.reshape(coherent_state, (-1,1))
    return coherent_state



#Returns the inner product of two states with an OpenFermion QubitOperator
def inner_product(qubit_op, state1, state2=None):
    qubit_op.compress()
    if state2 is None:
        state2 = state1.copy()
    if qubit_op == QubitOperator():
        return 0
    else:
        array_op = get_sparse_operator(qubit_op).toarray()
        return np.dot(state1.conj().T, np.dot(array_op, state2))[0][0]



#Finds minimum value of a mixed discrete-continuous symengine expression
def minimize_expr(expr, angle_folds, amplitude_folds, sampler, max_cycles=5, num_samples=1000, strength=1e3, verbose=False):
    
    #get lists of continuous and discrete variables
    try:
        all_vars = list(expr.free_symbols)
    except AttributeError:
        return expr
    disc_vars, cont_vars = sort_mixed_vars(all_vars)
    cont_bounds = get_bounds(cont_vars, angle_folds=angle_folds, amplitude_folds=amplitude_folds)
    disc_vals = np.random.choice([-1,1], size=len(disc_vars))
    cont_vals = np.random.uniform(low=cont_bounds.transpose()[0], high=cont_bounds.transpose()[1], size=len(cont_vars))
    
    all_disc_vals, all_cont_vals = [], []

    #minimize expression
    min_energies, cycle = [], 0
    for cycle in range(max_cycles):
        
        #minimize continuous variables for fixed discrete variables
        cont_expr = expr
        for i in range(len(disc_vars)):
            cont_expr = cont_expr.subs(disc_vars[i], disc_vals[i])
        
        f = se.lambdify(cont_vars, (cont_expr,))
        def g(x): return f(*x)
        results = sci.optimize.minimize(g, cont_vals, method='L-BFGS-B', bounds=cont_bounds)
        cont_vals = results.x

        #minimize discrete variables for fixed continuous variables
        disc_expr = expr
        for i in range(len(cont_vars)):
            disc_expr = disc_expr.subs(cont_vars[i], cont_vals[i])
        
        disc_expr = se.expand(disc_expr)
        bqm = dimod.higherorder.utils.make_quadratic(expr_to_dict(disc_expr), strength, dimod.SPIN)
        qubo, constant = bqm.to_qubo()
        
        #run sampler
        response = sampler.sample_qubo(qubo,num_reads=num_samples)
        solutions = pd.DataFrame(response.data())
        minIndex = int(solutions[['energy']].idxmin())
        minEnergy = round(solutions['energy'][minIndex],12) + constant
        unredSolution = solutions['sample'][minIndex]

        for key in unredSolution:
            try:
                index = disc_vars.index(key)
            except ValueError:
                continue
            disc_vals[index] = 2*unredSolution[key]-1
        min_energies += [minEnergy]
        
        all_disc_vals += [disc_vals]
        all_cont_vals += [cont_vals]
        
        if verbose:
            print('Cycle:',cycle+1,'Energy:',minEnergy)
    
    min_energy = min(min_energies)
    index = min_energies.index(min_energy)
    cont_dict = dict(zip(cont_vars, all_cont_vals[index]))
    disc_dict = dict(zip(disc_vars, all_disc_vals[index]))
    
    return min_energy, cont_dict, disc_dict



#Calculate QMF energy and optimal bloch angles
def QMF(qubit_H, angle_folds, sampler, num_cycles=5, num_samples=1000, strength=1e3, verbose=False):
    
    n = count_qubits(qubit_H)
    expr = qubit_op_to_expr(qubit_H, angle_folds=angle_folds)
    
    QMF_energy, cont_dict, disc_dict = minimize_expr(expr, angle_folds, 0, sampler,
        max_cycles=num_cycles, num_samples=num_samples, strength=strength, verbose=verbose)


    for key in cont_dict:
        num = str(key)[3:]
        if str(key)[:3] == 'phi':
            if angle_folds == 3:
                try:
                    W = disc_dict[se.Symbol('W'+str(num))]
                    if W == -1:
                        cont_dict[key] = np.pi - cont_dict[key]
                except KeyError:
                    pass
            if angle_folds >= 2:
                try:
                    Q = disc_dict[se.Symbol('Q'+str(num))]
                    if Q == -1:
                        cont_dict[key] = 2*np.pi - cont_dict[key]
                except KeyError:
                    pass
                
        elif str(key)[:3] == 'the':
            if angle_folds >= 1:
                try:
                    Z = disc_dict[se.Symbol('Z'+str(num))]
                    if Z == -1:
                        cont_dict[key] = np.pi - cont_dict[key]
                except KeyError:
                    pass
    
    return QMF_energy, cont_dict



#Calculate QCC energy and optimal bloch angles and entangler amplitudes
def QCC(qubit_H, entanglers, angle_folds, amplitude_folds, sampler,
        num_cycles=5, num_samples=1000, strength=1e3, verbose=False):
    
    n, N_ent = count_qubits(qubit_H), len(entanglers)
    if N_ent == 0:
        QMF_energy, bloch_angles = QMF(qubit_H, angle_folds, sampler, num_cycles=num_cycles,
            num_samples=num_samples, strength=strength, verbose=verbose)
        return QMF_energy, bloch_angles
    
    expr = qubit_op_to_expr(qubit_H, angle_folds=angle_folds)

    
    #get coefficients
    if amplitude_folds == 0:
        alpha = [se.sin(se.Symbol('tau'+str(i))) for i in range(N_ent)]
        beta = [(1-se.cos(se.Symbol('tau'+str(i)))) for i in range(N_ent)]
    elif amplitude_folds == 1:
        alpha = [se.Symbol('F'+str(i))*se.sin(se.Symbol('tau'+str(i))) for i in range(N_ent)]
        beta = [(1-se.cos(se.Symbol('tau'+str(i)))) for i in range(N_ent)]
    else:
        alpha = [se.Symbol('F'+str(i))*se.sin(se.Symbol('tau'+str(i))) for i in range(N_ent)]
        beta = [(1-se.Symbol('G'+str(i))*se.cos(se.Symbol('tau'+str(i)))) for i in range(N_ent)]

    
    #calculate QCC transformation
    expr = 0
    entanglers_str = list(entanglers.keys())
    term_combs = list(itertools.product('abn', repeat=N_ent))
    for term_comb in term_combs:
        coeff, term = 1, qubit_H
        for i in range(len(term_comb)):
            char, P = term_comb[i], entanglers[entanglers_str[i]]
            
            if char == 'a':
                coeff *= alpha[i]
                term = -1j/2*commutator(term, P)
            elif char == 'b':
                coeff *= beta[i]
                term = 1/2*P*commutator(term, P)
                
        term = qubit_op_to_expr(term, angle_folds=angle_folds)
        expr += coeff*term


    #minimize QCC expression
    QCC_energy, cont_dict, disc_dict = minimize_expr(expr, angle_folds, amplitude_folds, sampler,
        max_cycles=num_cycles, num_samples=num_samples, strength=strength, verbose=verbose)


    #unfold continuous variables
    for key in cont_dict:
        num = str(key)[3:]
        if str(key)[:3] == 'phi':
            if angle_folds == 3:
                try:
                    W = disc_dict[se.Symbol('W'+str(num))]
                    if W == -1:
                        cont_dict[key] = np.pi - cont_dict[key]
                except KeyError:
                    pass
            if angle_folds >= 2:
                try:
                    Q = disc_dict[se.Symbol('Q'+str(num))]
                    if Q == -1:
                        cont_dict[key] = 2*np.pi - cont_dict[key]
                except KeyError:
                    pass
                
        elif str(key)[:3] == 'the':
            if angle_folds >= 1:
                try:
                    Z = disc_dict[se.Symbol('Z'+str(num))]
                    if Z == -1:
                        cont_dict[key] = np.pi - cont_dict[key]
                except KeyError:
                    pass
                
        else:
            if amplitude_folds == 2:
                try:
                    G = disc_dict[se.Symbol('G'+str(num))]
                    if G == 1:
                        cont_dict[key] = np.pi - cont_dict[key]
                except KeyError:
                    pass
            if amplitude_folds >= 1:
                try:
                    F = disc_dict[se.Symbol('F'+str(num))]
                    if F == -1:
                        cont_dict[key] = 2*np.pi - cont_dict[key]
                except KeyError:
                    pass
    
    return QCC_energy, cont_dict
