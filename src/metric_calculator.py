import numpy as np
import solution_utils
import importlib
from tqdm import tqdm

#When toyinh with the data, it seems that exp_fo_mult and 
#critical_multiplicative are significant outliers in terms of being very poor. 
#For simplicity, we go ahead and thow them out. Recip_Pure_sak was thrown out 
#by accident. But qualititively that won't be a problem

OUTCASTS = ['exp_fo_mult', 'critical_multiplicative']

APPROXIMANTS = ['old_approximant', 
                'critical_additive',
                'exp_first_order',
                'writeup_sinh_cos',
                'writeup_sinh_poly',
                'pure_sakiadis',
                'open_sakiadis',
                'trig_fo_sak',
                'poly_fo_sak',
                'trig_to_sak',
                'poly_to_sak',
                'trig_so_sak',
                'poly_so_sak', 
                'near_recip_oc_sak',
                'nr_oc_fo_poly_sak',
                'nr_oc_fo_trig_sak',
                'nr_oc_so_poly_sak',
                'nr_oc_so_trig_sak',
                'nr_oc_to_poly_sak',
                'nr_oc_to_trig_sak',
                'tanh_fo_mult', 
                'tanh_oc_fo_sak', 
                'tanh_nr_oc_fo_sak',
                'tanh_oc_so_sak',
                'tanh_nr_oc_so_sak',
                'tanh_oc_to_sak',
                'tanh_nr_oc_to_sak']
CASES = [(np.sqrt(2)+.001, .2), (np.sqrt(2)+.01, .2), (1.45, .2), (1.53, .1), (1.6, .1), (2, .1), (3, .1)]
NS = [(5, .2), (10, .2), (15, .2), (25, .2), (35, .2)]

TOTAL = len(CASES) * len(NS) * len(APPROXIMANTS)

#import the approximant functions
APPROX_FUNCTS = {}
for approx in APPROXIMANTS:
    approx_module = importlib.import_module(approx)
    approx_funct = approx_module.get_approximant
    APPROX_FUNCTS[approx] = approx_funct

scores = {a : 0 for a in APPROXIMANTS}

pbar = tqdm(total=TOTAL)

for (W0, CASE_WEIGHT) in CASES:
    for (N, N_WEIGHT) in NS:
        #first, create the conditions of the problem. 
        m_conds = solution_utils.Conditions(0, W0, 20, 10, solution_utils._half_period(W0, 10, 20), .0001, 200, N)
        m_weight = CASE_WEIGHT * N_WEIGHT
        #find the numerical (exact, for us) solution.
        times, thetas = solution_utils.numerical_solve(m_conds)
        #now, let's find the L2 error for each approximant. 
        errors = {}
        for (approx, af) in APPROX_FUNCTS.items():
            m_af = af(m_conds)
            values = np.array([m_af(t) for t in times])
            error = np.linalg.norm(values - thetas)
            errors[approx] = error
            pbar.update(1)
        #aggregate the error data. 
        raw_errors = np.array(list(errors.values()))
        mu = np.mean(raw_errors)
        sigma = np.std(raw_errors)
        #now iterate over again and add it to the error. 
        for approx in APPROXIMANTS:
            score = m_weight * (errors[approx] - mu) / (sigma if sigma > 1e-06 else 1) 
            scores[approx] = scores[approx] + score
pbar.close()

print(scores)
print('\n\n\n')

SORTED_APPROXIMANTS = sorted(list(scores.items()), key = lambda x : -x[1])
print(SORTED_APPROXIMANTS)
        
'''
RESULT
---------------------

Finished in 30:08. 

{'old_approximant': -0.26281402625342104, 'critical_additive': 1.9288829685760671, 'exp_first_order': -0.1693849965214163, 'writeup_sinh_cos': 0.28712699784853113, 'writeup_sinh_poly': -0.31289498899830337, 'pure_sakiadis': -0.05503161014441993, 'open_sakiadis': -0.25215907811687077, 'trig_fo_sak': -0.23792886260513474, 'poly_fo_sak': -0.2327284547386652, 'trig_to_sak': -0.28175492469036334, 'poly_to_sak': -0.3112484175471845, 'trig_so_sak': -0.28839796855905564, 'poly_so_sak': -0.3207117636190678, 'near_recip_oc_sak': 0.7331439502083525, 'nr_oc_fo_poly_sak': -0.3105169235632155, 'nr_oc_fo_trig_sak': -0.29912294950108514, 'nr_oc_so_poly_sak': -0.30439981856934345, 'nr_oc_so_trig_sak': -0.2051317669689859, 'nr_oc_to_poly_sak': -0.32786276368168676, 'nr_oc_to_trig_sak': 0.5136562835319571, 'tanh_fo_mult': 2.3904803886607464, 'tanh_oc_fo_sak': -0.2983306933354529, 'tanh_nr_oc_fo_sak': -0.25216510652503027, 'tanh_oc_so_sak': -0.3170155216987336, 'tanh_nr_oc_so_sak': -0.21354149727953134, 'tanh_oc_to_sak': -0.3170155216987336, 'tanh_nr_oc_to_sak': -0.2831329342099523}




[('tanh_fo_mult', 2.3904803886607464), ('critical_additive', 1.9288829685760671), ('near_recip_oc_sak', 0.7331439502083525), ('nr_oc_to_trig_sak', 0.5136562835319571), ('writeup_sinh_cos', 0.28712699784853113), ('pure_sakiadis', -0.05503161014441993), ('exp_first_order', -0.1693849965214163), ('nr_oc_so_trig_sak', -0.2051317669689859), ('tanh_nr_oc_so_sak', -0.21354149727953134), ('poly_fo_sak', -0.2327284547386652), ('trig_fo_sak', -0.23792886260513474), ('open_sakiadis', -0.25215907811687077), ('tanh_nr_oc_fo_sak', -0.25216510652503027), ('old_approximant', -0.26281402625342104), ('trig_to_sak', -0.28175492469036334), ('tanh_nr_oc_to_sak', -0.2831329342099523), ('trig_so_sak', -0.28839796855905564), ('tanh_oc_fo_sak', -0.2983306933354529), ('nr_oc_fo_trig_sak', -0.29912294950108514), ('nr_oc_so_poly_sak', -0.30439981856934345), ('nr_oc_fo_poly_sak', -0.3105169235632155), ('poly_to_sak', -0.3112484175471845), ('writeup_sinh_poly', -0.31289498899830337), ('tanh_oc_so_sak', -0.3170155216987336), ('tanh_oc_to_sak', -0.3170155216987336), ('poly_so_sak', -0.3207117636190678), ('nr_oc_to_poly_sak', -0.32786276368168676)]

'''
