import argparse
import NaiveDE
import pandas as pd
import numpy as np
import gpflow as gpf
import time
import pickle
from sklearn.model_selection import train_test_split
import time


# Get command line arguments
parser = argparse.ArgumentParser(description='Saves the mean and variance for all noisy genes as dictionary')
parser.add_argument('-e','--expression', help='Path of the expression matrix to process. Rows are spots, columns are gene expression. First column should be barcode or index.', type=str, required=True)
parser.add_argument('-n', '--normalize', help='Whether to normalize (and variance stablizied) the expression.', action='store_true')
parser.add_argument('-l','--locations', help='Path of the spot locations. Rows are spots, columns are spot locations. Number of rows must be same as expression matrix, number of columns = 2', type=str)
parser.add_argument('-i','--input', help='Path of the csv file to saved in step 2', type=str, required=True)
parser.add_argument('-o','--output', help='Path of the dictionary file containing modelled gene and variance for the noisy genes', type=str, required=True)


args = vars(parser.parse_args())


df = pd.read_csv(args["expression"], index_col=0)

# Normalize
if args.normalized:
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes
    df = NaiveDE.stabilize(df.T).T
    samp_info = pd.DataFrame(df.sum(1), columns=["total_counts"])
    df = NaiveDE.regress_out(samp_info, df.T, 'np.log(total_counts)').T

genes = df.columns[:-2]
genes = np.sort(genes)

X = pd.DataFrame(args["locations"]).values
X = X.astype(np.float64)
N = X.shape[0]

df_result = pd.read_csv(args["input"]).set_index("gene")
df_sig = df_result[ (df_result.q_wilcox_test < 0.05) & (df_result.nlpd_homo_test > df_result.nlpd_hetero_test) ]
genes_sig = df_sig.index.values


Y_all_train = []
Y_mean_all_train = []
Y_var_all_train = []

                 
HGP_mean_all_train = []
HGP_var_all_train = []
                 
                 

eps = 1e-8
N = X.shape[0]

                 
start_time = time.time()
n_genes = genes_sig.shape[0]

for i, gene in enumerate(genes_sig):

    k = i + 1
    if i%100 == 0:
        print(f"Currently running for gene {gene} ({k}/{n_genes})")
        print(f"Time taken for last 50 genes: {time.time() - start_time} seconds")
        start_time = time.time()
    
    
    Y = df[gene].values.reshape(-1, 1)
    
    

    model_homo = gpf.models.GPR(
        data=(X, Y),
        mean_function=gpf.mean_functions.Constant(),
        kernel=gpf.kernels.SquaredExponential()
    )


    opt = gpf.optimizers.Scipy()
    opt_logs = opt.minimize(model_homo.training_loss, model_homo.trainable_variables, options=dict(maxiter=200))



    Ymean_train, Yvar_train = model_homo.predict_y(X)
    HGP_mean_train = Ymean_train.numpy().copy()



    z = (Y - Ymean_train.numpy())**2 - Yvar_train.numpy()



    model_hetero = gpf.models.GPR(
        data=(X, z),
        mean_function=None,
        kernel=gpf.kernels.SquaredExponential()
        )

    opt_2 = gpf.optimizers.Scipy()
    opt_logs_2 = opt_2.minimize(model_hetero.training_loss, model_hetero.trainable_variables, options=dict(maxiter=500))

    #ll = model_hetero.maximum_log_likelihood_objective()
    Vmean_train, Vvar_train = model_hetero.predict_y(X)



    HGP_var_train = Yvar_train.numpy() + Vmean_train.numpy()
    HGP_var_train[HGP_var_train < 1e-12] = 1e-12
 
    # LOG
    Y_all_train.append(Y)
    Y_mean_all_train.append(Ymean_train.numpy())
    Y_var_all_train.append(Yvar_train.numpy)


    HGP_mean_all_train.append(Ymean_train.numpy())
    HGP_var_all_train.append(HGP_var_train)



Y_all_train = Y_all_train.squeeze()
Y_mean_all_train = Y_mean_all_train.squeeze()
Y_var_all_train = Y_var_all_train.squeeze()
HGP_mean_all_train = HGP_mean_all_train.squeeze()
HGP_var_all_train = HGP_var_all_train.squeeze()

d = {}
for i, gene in enumerate(genes_sig):
    d[gene] = {}
    d[gene]['exp'] = Y_all_train[i]
    d[gene]['mean'] = Y_mean_all_train[i]
    d[gene]['var_homo'] = Y_var_all_train[i]
    d[gene]['var_hetero'] = HGP_var_all_train[i]

with open(args["output"], 'wb') as f:
        pickle.dump(d, f)
