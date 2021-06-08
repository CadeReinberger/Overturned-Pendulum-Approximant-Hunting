import re
import pandas as pd

measured_approximants = [('tanh_fo_mult', 2.3904803886607464), ('criticial_additive', 1.9288829685760671), ('near_recip_oc_sak', 0.7331439502083525), ('nr_oc_to_trig_sak', 0.5136562835319571), ('writeup_sinh_cos', 0.28712699784853113), ('pure_sakiadis', -0.05503161014441993), ('exp_first_order', -0.1693849965214163), ('nr_oc_so_trig_sak', -0.2051317669689859), ('tanh_nr_oc_so_sak', -0.21354149727953134), ('poly_fo_sak', -0.2327284547386652), ('trig_fo_sak', -0.23792886260513474), ('open_sakiadis', -0.25215907811687077), ('tanh_nr_oc_fo_sak', -0.25216510652503027), ('old_approximant', -0.26281402625342104), ('trig_to_sak', -0.28175492469036334), ('tanh_nr_oc_to_sak', -0.2831329342099523), ('trig_so_sak', -0.28839796855905564), ('tanh_oc_fo_sak', -0.2983306933354529), ('nr_oc_fo_trig_sak', -0.29912294950108514), ('nr_oc_so_poly_sak', -0.30439981856934345), ('nr_oc_fo_poly_sak', -0.3105169235632155), ('poly_to_sak', -0.3112484175471845), ('writeup_sinh_poly', -0.31289498899830337), ('tanh_oc_so_sak', -0.3170155216987336), ('tanh_oc_to_sak', -0.3170155216987336), ('poly_so_sak', -0.3207117636190678), ('nr_oc_to_poly_sak', -0.32786276368168676)]
measured_approximants = measured_approximants[::-1]

approximants_error = dict(measured_approximants)

def read_table_source_df():
    res = None
    with open('first_table_simplified.txt', 'r') as f:
        fulltxt = re.sub('\n', '', ''.join(f.readlines()))
        table = [[e.strip() for e in line.split(r'&')] for line in re.compile(r'\\\s+\\hline').split(fulltxt)]
        res = pd.DataFrame(table)
        res = res.drop(res.index[-1])
    return res

def check_table_source_excel():
    read_table_source_df().to_excel('testdoc.xlsx')
    
def de_texify_filename(filename):
    filename = re.sub(r'\\_', r'_', filename)
    filename = filename[:filename.index('.')]
    return filename

def get_cleaned_df():
    df = read_table_source_df()
    df[3] = [approximants_error[de_texify_filename(d)] if de_texify_filename(d) in approximants_error else None for d in df[2]]
    df = df.sort_values(by=3)
    df = df.dropna()
    return df

def get_cleaned_text():
    cdf = get_cleaned_df()
    clean_table = cdf.values.tolist()
    res_txt = r' \\ \hline '.join([' & '.join([str(a).strip() for a in line]) for line in clean_table])    
    start = r'\begin{tabular}{|R{.12}|R{.53}|R{.15}|R{.2}|} \hline Description & Equation & Python Filename & Qualitative Description \\ \hline '
    end = r'\end{tabular}'
    tot = start + res_txt + end
    return tot
    
print(get_cleaned_text())        
