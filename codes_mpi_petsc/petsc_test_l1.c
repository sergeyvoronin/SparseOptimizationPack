static char help[] = "Reads in a petsc matrix and vector and tests the Landweber/Fista iteration algorithms";

/* 
read in matrix A and vector b
tries to recover x using thresholded Landweber/Fista iteration 
svoronin */

#include "petscksp.h"
#include <stdlib.h>
#include <string.h>

/* extra includes for rand() */
#include <stdlib.h>
#include <time.h>

/* include l1-opt functions */
#include "petsc_l1/l1optmethods.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
    Mat A;
    Vec b,x,x0,dn,z,v;
    PetscViewer    pv;               /* viewers are needed for file i/o ops */
    PetscErrorCode ierr ;
    PetscInt       i,m,n,mloc,nloc,num_taus,numiters,maxiters,use_wavelets = 1; 
    PetscScalar    tau, tau_min, tau_max, tau_step, tau_optimal, TOL;
    PetscScalar    *residuals, *Js;
    PetscReal      rval, norm_val, norm_val1, norm_val2, percent_error, min_percent_error;
    PetscMPIInt    rank, size;
    char mat_file[PETSC_MAX_PATH_LEN];
    char vecb_file[PETSC_MAX_PATH_LEN];
    char output_file[PETSC_MAX_PATH_LEN];
    FILE *fp;

    /* initialize petsc */
    PetscInitialize(&argc,&args,(char *)0,help);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    PetscPrintf(PETSC_COMM_SELF,"Number of processors = %d, rank = %d\n",size,rank);	

    /* read command line arguments */
    ierr = PetscOptionsGetString(PETSC_NULL,"-mat_file",mat_file,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(PETSC_NULL,"-vecb_file",vecb_file,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(PETSC_NULL,"-output_file",output_file,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);

    /* read in A and b from disk */
    PetscPrintf(PETSC_COMM_WORLD,"Loading matrix A from disk...\n");	
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat_file,FILE_MODE_READ,&pv);CHKERRQ(ierr);
    ierr = MatLoad(pv,MATAIJ,&A);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(pv);CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD,"Loading vector b from disk...\n");	
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecb_file,FILE_MODE_READ,&pv);CHKERRQ(ierr);
    ierr = VecLoad(pv,VECMPI,&b);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(pv);CHKERRQ(ierr);

    /* get mat sizes */
    ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&mloc,&nloc);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"A global sizes: %d by %d\n", m,n);
    PetscPrintf(PETSC_COMM_SELF,"A local sizes (rank %d): %d by %d\n", rank,mloc,nloc);

    /* vec get size */   
    ierr = VecGetSize(b,&m); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"b global size: %d\n", m);

    /* get mat norm and scale A,b to make ||A||_2 < 1 */
    MatNorm(A,NORM_FROBENIUS,&norm_val);
    PetscPrintf(PETSC_COMM_WORLD,"norm(A,FROBENIUS) = %f\n", norm_val);
    MatNorm(A,NORM_1,&norm_val);
    PetscPrintf(PETSC_COMM_WORLD,"norm(A,1) = %f\n", norm_val);
    ierr = getLargestSingularValue(A, 20, &norm_val);
    PetscPrintf(PETSC_COMM_WORLD,"norm(A,spectral) = %f\n", norm_val);

    // now scale so spectral norm < 1
    if(norm_val >= 1){
        PetscPrintf(PETSC_COMM_WORLD, "Scaling system\n");
        ierr = MatScale(A, 1/(2*norm_val)); CHKERRQ(ierr);
        ierr = VecScale(b, 1/(2*norm_val)); CHKERRQ(ierr); 
    }

    /* create vectors */
    VecCreate(PETSC_COMM_WORLD,&x);
    VecSetSizes(x,PETSC_DECIDE,n);
    VecSetFromOptions(x);
    VecCreate(PETSC_COMM_WORLD,&dn);
    VecSetSizes(dn,PETSC_DECIDE,m);
    VecSetFromOptions(dn);

    /* generate initial guess x0 = 0 */
    VecCreate(PETSC_COMM_WORLD,&x0);
    VecSetSizes(x0,PETSC_DECIDE,n);
    VecSetFromOptions(x0);
    ierr = VecSet(x0,0.0); CHKERRQ(ierr);

    /* calculate z = A^{t} * b or W*(A^{t} * b) */    
    VecCreate(PETSC_COMM_WORLD,&v);
    VecSetSizes(v,PETSC_DECIDE,n);
    VecSetFromOptions(v);
    VecCreate(PETSC_COMM_WORLD,&z);
    VecSetSizes(z,PETSC_DECIDE,n);
    VecSetFromOptions(z);

    ierr = MatMultTranspose(A,b,v); CHKERRQ(ierr);
    VecMax(v,&i,&tau_max);
    VecNorm(v,NORM_2,&rval);
    PetscPrintf(PETSC_COMM_WORLD, "max( (A^{t} * b) = %f\n", tau_max);
    PetscPrintf(PETSC_COMM_WORLD, "norm( (A^{t} * b) , 2 ) = %f\n", rval);

    if(use_wavelets == 0){
        VecCopy(v,z);
        VecMax(z,&i,&tau_max);
        VecNorm(z,NORM_2,&rval);
    }
    else if(use_wavelets == 1){
        ierr = vecWT(v,&z,"inversetranspose"); CHKERRQ(ierr);
        VecMax(z,&i,&tau_max);
        VecNorm(z,NORM_2,&rval);
        PetscPrintf(PETSC_COMM_WORLD, "max( W*(A^{t} * b) = %f\n", tau_max);
        PetscPrintf(PETSC_COMM_WORLD, "norm( W*(A^{t} * b) , 2 ) = %f\n", rval);
    }
    else{
        PetscPrintf(PETSC_COMM_WORLD,"invalid wavelet option, exiting.\n");
        return -1;
    }
    
    PetscPrintf(PETSC_COMM_WORLD, "\n test projection onto l1 ball..\n");
    ierr = projectOnL1Ball(b, &v, 50); CHKERRQ(ierr);
    VecNorm(b,NORM_2,&rval);
    PetscPrintf(PETSC_COMM_WORLD,"---> norm(b,2) = %f\n", rval);
    VecNorm(v,NORM_2,&rval);
    PetscPrintf(PETSC_COMM_WORLD,"---> norm(v,2) = %f\n", rval);


    // call Landweber iteration --->
    tau = tau_max/10;
    //tau = 0.005;
    maxiters = 10;
    TOL = 1e-10;
    residuals = (PetscScalar*)malloc((maxiters+1)*sizeof(PetscScalar));
    Js = (PetscScalar*)malloc((maxiters+1)*sizeof(PetscScalar));

    PetscPrintf(PETSC_COMM_WORLD,"running l1 iteration with tau = %f (tau_max = %f)..\n", tau, tau_max);
    if(use_wavelets == 0){
        //ierr = hardThresholdedLandweber(A, b, tau, x0, TOL, maxiters, &x, &numiters,residuals,Js); CHKERRQ(ierr);
        //ierr = thresholdedLandweber(A, b, tau, x0, TOL, maxiters, &x, &numiters,residuals,Js); CHKERRQ(ierr);
        ierr = thresholdedFista(A, b, tau, x0, TOL, maxiters, &x, &numiters,residuals,Js); CHKERRQ(ierr);
    }
    else if(use_wavelets == 1){
        ierr = thresholdedFistaWT(A, b, tau, x0, TOL, maxiters, &x, &numiters,residuals,Js); CHKERRQ(ierr);
    }
    PetscPrintf(PETSC_COMM_WORLD,"finished with numiters = %d\n", numiters);
   
    /* calculate norms and residual */ 
    PetscPrintf(PETSC_COMM_WORLD,"tau_max = %f\n", tau_max);
    PetscPrintf(PETSC_COMM_WORLD,"tau = %f\n", tau);
    ierr = VecNorm(x,NORM_2,&norm_val); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"norm(x,2) = %f\n", norm_val);
    ierr = MatMult(A,x,dn); CHKERRQ(ierr); //dn=A*x
    ierr = VecAXPY(dn,-1,b); CHKERRQ(ierr); //dn = -b + dn = A*x - b
    ierr = VecNorm(dn,NORM_2,&norm_val); CHKERRQ(ierr); //||A*x-b||
    PetscPrintf(PETSC_COMM_WORLD,"||A*x-b|| = %f\n", norm_val);

    /* write residuals, Js, and runinfo to disk */
    if(rank == 0){
        PetscPrintf(PETSC_COMM_SELF,"writing residuals and Js to disk..\n");
        fp = fopen("data/petsc_test_l1/residuals.txt","w");
        for(i=0; i<=numiters; i++){
           fprintf(fp,"%f\n",residuals[i]);
        }
        fclose(fp);

        fp = fopen("data/petsc_test_l1/Js.txt","w");
        for(i=0; i<=numiters; i++){
           fprintf(fp,"%f\n",Js[i]);
        }
        fclose(fp);

        fp = fopen("data/petsc_test_l1/run_info.txt","w");
        fprintf(fp, "run with vecb_file = %s\n", vecb_file);
        fprintf(fp, "numiters = %d\n", numiters);
        fclose(fp);
    }

    /* write solution x to disk */
    PetscPrintf(PETSC_COMM_WORLD,"Writing solution x to %s.\n", output_file);	
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,output_file,FILE_MODE_WRITE,&pv);CHKERRQ(ierr);
    ierr = VecView(x,pv); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(pv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"done writing.\n");	

   
    /* destroy the mat and vecs */
    PetscPrintf(PETSC_COMM_WORLD,"Freeing up memory by destroying the matrix and vectors.\n");
    ierr = MatDestroy(A); CHKERRQ(ierr);
    ierr = VecDestroy(b); CHKERRQ(ierr);
    ierr = VecDestroy(x); CHKERRQ(ierr);
    ierr = VecDestroy(dn); CHKERRQ(ierr);
    ierr = VecDestroy(v); CHKERRQ(ierr);
    ierr = VecDestroy(z); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Done freeing memory, exiting.\n");

    /* finalize and exit */
    ierr = PetscFinalize();CHKERRQ(ierr);
    return 0;
}

