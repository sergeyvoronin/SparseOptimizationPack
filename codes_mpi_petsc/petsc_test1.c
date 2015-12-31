static char help[] = "Reads in a petsc matrix and vector generated previously with codes in make_matrix/";

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
//#include "petsc_l1/l1optmethods.h"


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
    PetscBool      flg;
    char* mat_file = (char*)malloc(sizeof(char)*PETSC_MAX_PATH_LEN);
    char* vecb_file = (char*)malloc(sizeof(char)*PETSC_MAX_PATH_LEN);
    char* output_file = (char*)malloc(sizeof(char)*PETSC_MAX_PATH_LEN);
    FILE *fp;

    /* initialize petsc */
    PetscInitialize(&argc,&args,(char *)0,help);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    PetscPrintf(PETSC_COMM_SELF,"Number of processors = %d, rank = %d\n",size,rank);	

    /* read command line arguments */
    ierr = PetscOptionsGetString(PETSC_NULL,"-mat_file",mat_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"error setting mat file location");
    ierr = PetscOptionsGetString(PETSC_NULL,"-vecb_file",vecb_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"error setting vec file location");
    ierr = PetscOptionsGetString(PETSC_NULL,"-output_file",output_file,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"error setting output file location");
    


    /* read in A and b from disk */
    PetscPrintf(PETSC_COMM_WORLD,"Loading matrix A from disk: %s\n", mat_file);
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&pv); CHKERRQ(ierr);
    ierr = PetscViewerSetType(pv, PETSCVIEWERBINARY); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,mat_file,FILE_MODE_READ,&pv);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetType(A,MATSEQAIJ); CHKERRQ(ierr);
    ierr = MatLoad(A,pv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&pv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"done reading.\n");

    PetscPrintf(PETSC_COMM_WORLD,"Loading vector b from disk: %s\n", vecb_file);	
    ierr = PetscViewerCreate(PETSC_COMM_WORLD,&pv);
    ierr = PetscViewerSetType(pv, PETSCVIEWERBINARY); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecb_file,FILE_MODE_READ,&pv);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
    ierr = VecLoad(b,pv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&pv);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"done reading.\n");


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
    VecNorm(b,NORM_2,&norm_val);
    PetscPrintf(PETSC_COMM_WORLD,"norm(b,2) = %f\n", norm_val);
    //ierr = getLargestSingularValue(A, 20, &norm_val);
    //PetscPrintf(PETSC_COMM_WORLD,"norm(A,spectral) = %f\n", norm_val);

    /* destroy the mat and vecs */
    PetscPrintf(PETSC_COMM_WORLD,"Freeing up memory by destroying the matrix and vectors.\n");
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Done freeing memory, exiting.\n");

    /* finalize and exit */
    ierr = PetscFinalize();CHKERRQ(ierr);
    return 0;
}

