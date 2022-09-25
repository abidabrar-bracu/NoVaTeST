import gpflow
import numpy as np
import gpflow as gpf
import pandas as pd
import time
import glob
import pickle
from sklearn.model_selection import train_test_split
import time
import os
import argparse
import NaiveDE

# Get command line arguments
parser = argparse.ArgumentParser(description='Models all the genes and saves the results and the models for further processing')

parser.add_argument('-e','--expression', help='Path of the expression matrix to process. Rows are spots, columns are gene expression. First column should be barcode or index.', type=str, required=True)
parser.add_argument('-n', '--normalize', help='Whether to normalize (and variance stablizied) the expression.', action='store_true')
parser.add_argument('-l','--locations', help='Path of the spot locations. Rows are spots, columns are spot locations. Number of rows must be same as expression matrix, number of columns = 2', type=str)
parser.add_argument('-r','--resume', help="Whether to resume from the previous gene in the saved directory or start from the beginning", action='store_true')
parser.add_argument('-o','--odir', help='Output directory to save the variables and the models', type=str, required=True)

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
n_genes = genes.shape[0]

X = pd.DataFrame(args["locations"]).values
X = X.astype(np.float64)

# Create output directories
save_dir_head = args["odir"]
folders = ["models", "variables"]


for f in folders:
    if not os.path.exists( os.path.join(save_dir_head, f) ):
        os.mkdir( os.path.join(save_dir_head, f) )

for f in folders:
    if not os.path.exists( os.path.join(save_dir_head, f, "homo") ):
        os.mkdir( os.path.join(save_dir_head, f, "homo") )
    if not os.path.exists( os.path.join(save_dir_head, f, "hetero") ):
        os.mkdir( os.path.join(save_dir_head, f, "hetero") )

save_dir_model = os.path.join(args["odir"], "models")
save_dir_variables = os.path.join(args["odir"], "variables")

if args.resume:
    resume = max(0, min([len(glob.glob( os.path.join(save_dir_head, folder) + "/homo/*")) for folder in folders])//10)
    print(f"Resuming from: {resume}")
else:
    resume = 0
    print(f"Starting from: {resume}")

eps = 1e-8
N = X.shape[0]

                 
start_time = time.time()
for i, gene in enumerate(genes[resume:]):
    k = i + resume + 1
    if i%10 == 0:
        print(f"Currently running for gene {gene} ({k}/{n_genes})")
        print(f"Time taken for last 10 genes: {time.time() - start_time} seconds")
        start_time = time.time()
    
    
    Y = df[gene].values.reshape(-1, 1)
    
    
    for i_kf in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]

        model_homo = gpflow.models.GPR(
            data = (X_train, Y_train),
            mean_function = gpflow.mean_functions.Constant(),
            kernel = gpf.kernels.SquaredExponential()
        )


        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(model_homo.training_loss, model_homo.trainable_variables, options=dict(maxiter=200))


        Ymean_train, Yvar_train = model_homo.predict_y(X_train)
        HGP_mean_train = Ymean_train.numpy().copy()
        mse_homo_train = np.sum(((Y_train - Ymean_train.numpy())**2)/Yvar_train.numpy())/N_train
        nlpd_homo_train = np.sum( np.log(2*np.pi*(Yvar_train)) + ((Y_train - Ymean_train)**2)/(Yvar_train) )/(2*N_train)


        Ymean_test, Yvar_test = model_homo.predict_y(X_test)
        HGP_mean_test = Ymean_test.numpy().copy()
        mse_homo_test = np.sum(((Y_test - Ymean_test.numpy())**2)/Yvar_test.numpy())/N_test
        nlpd_homo_test = np.sum( np.log(2*np.pi*(Yvar_test)) + ((Y_test - Ymean_test)**2)/(Yvar_test) )/(2*N_test)


        # SAVE THE VARIABLES FOR REPRODUCIBILITY
        filename = save_dir_variables + f"/homo/{k:04d}_{i_kf}_{gene}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump([Y_train, Y_test, 
                Ymean_train.numpy(), Yvar_train.numpy(), mse_homo_train, nlpd_homo_train,
                Ymean_test.numpy(), Yvar_test.numpy(), mse_homo_test, nlpd_homo_test], f)

        # SAVE THE MODEL FOR REPRODUCIBILITY
        filename = save_dir_model + f"/homo/{k:04d}_{i_kf}_{gene}.pickle"
        params = gpflow.utilities.parameter_dict(model_homo)
        with open(filename, 'wb') as fp:
            pickle.dump(params, fp)



        z = (Y_train - Ymean_train.numpy())**2 - Yvar_train.numpy()



        model_hetero = gpflow.models.GPR(
            data = (X_train, z),
            mean_function = None,
            kernel = gpf.kernels.SquaredExponential()
        )

        opt_2 = gpflow.optimizers.Scipy()
        opt_logs_2 = opt_2.minimize(model_hetero.training_loss, model_hetero.trainable_variables, options=dict(maxiter=500))

        #ll = model_hetero.maximum_log_likelihood_objective()
        Vmean_train, Vvar_train = model_hetero.predict_y(X_train)
        Vmean_test, Vvar_test = model_hetero.predict_y(X_test)




        HGP_var_train = Yvar_train.numpy() + Vmean_train.numpy()
        HGP_var_train[HGP_var_train < 1e-12] = 1e-12
        mse_hetero_train = np.sum(((Y_train - HGP_mean_train)**2)/(HGP_var_train))/N_train
        nlpd_hetero_train = np.sum( np.log(2*np.pi*(HGP_var_train)) + 
                                ((Y_train - Ymean_train.numpy())**2)/(HGP_var_train) )/(2*N_train)

        HGP_var_test = Yvar_test.numpy() + Vmean_test.numpy()
        HGP_var_test[HGP_var_test < 1e-12] = 1e-12
        mse_hetero_test = np.sum(((Y_test - HGP_mean_test)**2)/(HGP_var_test))/N_test
        nlpd_hetero_test = np.sum( np.log(2*np.pi*(HGP_var_test)) + 
                                ((Y_test - Ymean_test.numpy())**2)/(HGP_var_test) )/(2*N_test)

        

        # SAVE THE VARIABLES FOR REPRODUCIBILITY
        filename = save_dir_variables + f"/hetero/{k:04d}_{i_kf}_{gene}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump([z, Vmean_train.numpy(),
                        HGP_mean_train, HGP_var_train, mse_hetero_train, nlpd_hetero_train, 
                        HGP_mean_test, HGP_var_test, mse_hetero_test, nlpd_hetero_test], f)

        # SAVE THE MODEL FOR REPRODUCIBILITY
        filename = save_dir_model + f"/hetero/{k:04d}_{i_kf}_{gene}.pickle"
        params = gpflow.utilities.parameter_dict(model_hetero)
        with open(filename, 'wb') as fp:
            pickle.dump(params, fp)
