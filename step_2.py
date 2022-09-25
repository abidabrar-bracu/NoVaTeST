import glob
from scipy.stats import wilcoxon
import tqdm
import pickle
import argparse
import os
import numpy as np
from p_tqdm import p_map
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection


parser = argparse.ArgumentParser(description='Calculates p-values')
parser.add_argument('-i','--input', help='Directory of the saved variables', type=str, required=True)
parser.add_argument('-o','--output', help='Path of the csv file to save the final result', type=str, required=True)

args = vars(parser.parse_args())


save_dir_head = args["input"]
save_dir_homo = os.path.join(save_dir_head, "homo")
save_dir_hetero = os.path.join(save_dir_head, "hetero")

def process(gene):
    to_return = []
    to_return.append(gene)
    homo_files = glob.glob(save_dir_homo + f"/*_{gene}.pickle")
    homo_files = np.sort(homo_files)
    
    hetero_files = glob.glob(save_dir_hetero + f"/*_{gene}.pickle")
    hetero_files = np.sort(hetero_files)
    
    
    try:
        nlpd_test_homo_arr = []
        for hom in homo_files:
            with open(hom, 'rb') as f:
                vals_homo = pickle.load(f)
            _, _, _, _, _, _, _, _, _, nlpd_homo_test = vals_homo

            nlpd_test_homo_arr.append(nlpd_homo_test)

        to_return.append(np.mean(nlpd_test_homo_arr))



        nlpd_test_hetero_arr = []
        for het in hetero_files:
            with open(het, 'rb') as f:
                vals_hetero = pickle.load(f)
            _, _, _, _, _, _, _, _, _, nlpd_hetero_test = vals_hetero
           
            nlpd_test_hetero_arr.append(nlpd_hetero_test)

        to_return.append(np.mean(nlpd_test_hetero_arr))



        _, p_test = wilcoxon(nlpd_test_homo_arr, nlpd_test_hetero_arr)
        to_return.append(p_test)
        
        return to_return
    except:
        print(f"error with gene - {gene}")
        to_return.append("error")
        return to_return


genes_dir = glob.glob(save_dir_homo + "/*.pickle")
genes = [g.split("/")[-1].split("_")[-1][:-7] for g in genes_dir]

results = p_map(process, genes, num_cpus=8)


survived_genes = []
nlpd_homo_test_all = []
nlpd_hetero_test_all = []
pval_wilcox_test = []


for result in tqdm(results):
    if "error" in result:
        continue
    
    survived_genes.append(result.pop(0))
    nlpd_homo_test_all.append(result.pop(0))
    nlpd_hetero_test_all.append(result.pop(0))
    pval_wilcox_test.append(result.pop(0))
 
import pandas as pd


genes_selected = survived_genes#genes[genes != genes[451]]


_, qvals_wilcox_test = fdrcorrection(np.array(pval_wilcox_test), alpha=0.05, method='indep', is_sorted=False)

    
test_dict = {"gene":genes_selected,
       "nlpd_homo_test":nlpd_homo_test_all,
       "nlpd_hetero_test":nlpd_hetero_test_all,
       "pval_wilcox_test": pval_wilcox_test, 
       "q_wilcox_test": qvals_wilcox_test}

df = pd.DataFrame(test_dict).set_index("gene")

df.to_csv(args["output"])
