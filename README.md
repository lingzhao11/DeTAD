# DeTAD

***Decoupling Hi-C matrices with non-negative matrix factorization to reveal assorted TADs

Requirements enviorment : Python3.8 or above.

packages : including numpy, multiprocessing, sklearn, and scipy.

Usage python DeTAD.py -in Hi_C_matrix

#Parameters

-in : the input file of a N*N Hi-C matrix separated by TAB for a chromosome i.

-lambda : interaction strength for weighted regularization

-tau : strength of distance-aware regularization

-phi : scaling factor of distance-aware matrix

-psi : decay factor of distance-aware matrices

-max_iter : the iteration number of table H. (Default value is 1261)

-tol : Tolerance for convergence

-out : the output file for the predicted TAD of chromosome i, where a line represents a TAD containing two columns that represent the start bin and the end bin of a TAD.

contact: zhaoling2-c@my.cityu.edu.hk
