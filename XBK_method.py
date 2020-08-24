import pandas as pd
import dimod, math, itertools
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits

from helper_functions import *



#Applies XBK transformation to an OpenFermion QubitOperator
def XBK_transform(op, r, p):
    
    n = count_qubits(op)
    op_terms = op.terms
    new_op = QubitOperator()
    
    #transform operator term by term
    for key in op_terms:
        coeff = op_terms[key]
        term = QubitOperator()
        
        #cycle through each of the r ancillarly qubit copies
        for j in range(r):
            for k in range(r):
                sign = 1 if (j < p) == (k < p) else -1
                sub_term = QubitOperator('', 1)
                
                #cycle through each of the n original qubits
                spot = 0
                for i in range(n):
                    try:
                        if key[spot][0] == i:
                            char = key[spot][1]
                            spot += 1
                        else:
                            char = 'I'
                    except IndexError:
                        char = 'I'
                    
                    #use variable type to apply correct mapping
                    if char == 'X':
                        if j == k:
                            sub_term = QubitOperator('', 0)
                            break
                        else:
                            sub_term *= QubitOperator('', 1/2) - QubitOperator('Z'+str(i+n*j)+' Z'+str(i+n*k), 1/2)
                    elif char == 'Y':
                        if j == k:
                            sub_term = QubitOperator('', 0)
                            break
                        else:
                            sub_term *= QubitOperator('Z'+str(i+n*k), 1j/2) - QubitOperator('Z'+str(i+n*j), 1j/2)
                    elif char == 'Z':
                        if j == k:
                            sub_term *= QubitOperator('Z'+str(i+n*j), 1)
                        else:
                            sub_term *= QubitOperator('Z'+str(i+n*j), 1/2) + QubitOperator('Z'+str(i+n*k), 1/2)
                    else:
                        if j == k:
                            continue
                        else:
                            sub_term *= QubitOperator('', 1/2) + QubitOperator('Z'+str(i+n*j)+' Z'+str(i+n*k), 1/2)
                            
                term += sign*sub_term
        new_op += coeff*term
    
    new_op.compress()
    return new_op



#Construct C term required for XBK method
def construct_C(n, r, p):
    
    C = QubitOperator('', 0)
    perms = list(itertools.product([1,-1], repeat=n))
    
    for perm in perms:
        term = QubitOperator('', 0)
        
        for j in range(r):
            product = QubitOperator('', 1)
            
            for i in range(n):
                product *= QubitOperator('', 1/2) + QubitOperator('Z'+str(i+n*j), perm[i]/2)
                
            sign = -1 if j < p else 1
            term += sign*product
        C += term**2
    return C



#Find minimum energy and ground state using XBK method
def XBK(qubit_Hs, qubit_Cs, r, sampler, starting_lam=0, num_samples=1000, strength=1e3, verbose=False):
    
    n = count_qubits(qubit_Hs[0])
    min_energies, ground_states = [],[]
    
    for p in range(int(math.ceil(r/2+1))):
        qubit_H, qubit_C = qubit_Hs[p], qubit_Cs[p]
        
        
        #create C function to evalute sum(b^2)
        C_dict = convert_dict(qubit_C.terms)
        C_func = dict_to_func(C_dict)
        
        #calculate minimum energy for particular p value
        lam = starting_lam
        eigenvalue = min_energy = -1
        ground_state = []
        cycles = 0
        while min_energy < 0 and cycles < 10:
            #subtract lambda C from H
            H_prime = qubit_H - lam*qubit_C
            
            #construct qubo from reduced Hamiltonian
            bqm = dimod.higherorder.utils.make_quadratic(convert_dict(H_prime.terms), strength, dimod.SPIN)
            qubo, constant = bqm.to_qubo()
            
            if qubo == {}:
                break
            
            #run sampler
            response = sampler.sample_qubo(qubo,num_reads=num_samples)
            solutions = pd.DataFrame(response.data())

            #get mininum energy solution
            index = int(solutions[['energy']].idxmin())
            min_energy = round(solutions['energy'][index], 14) + constant
            full_solution = solutions['sample'][index]
            
            solution = []
            for key in ['s'+str(i) for i in range(n)]:
                try:
                    solution += [2*full_solution[key]-1]
                except KeyError:
                    solution += [0]
                    print('KeyError: '+str(key))
                    

            #calculate sum(b^2)
            try:
                sumBsq = int(C_func(*solution))
            except TypeError:
                sumBsq = int(C_func)

            if sumBsq == 0: #stop the loop in zero case
                cycles += 1
                break
            
            #calculate the eigenvalue of H
            eigenvalue = lam + min_energy/sumBsq

            #set lam equal to eigenvalue for next cycle
            if min_energy < 0:
                lam = eigenvalue
                ground_state = [(val+1)//2 for val in solution]
            cycles += 1
        
        min_energies += [round(lam, 14)]
        ground_states += [ground_state]
        
        if verbose:
            print('P:', p, 'E:', round(lam, 5))
        
    index = min_energies.index(min(min_energies))
    min_energy = min_energies[index]
    ground_state = ground_states[index]
    
    if verbose:
        print('Energy:', round(min_energy, 5))
    
    return min_energy, ground_state
