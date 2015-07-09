/* implements some iterative l1-optimization methods
svoronin
*/

#include "l1optmethods.h"

// define some globals --->
// monitoring
PetscInt monitor_flag = 0;
PetscInt monitor_save_number = 0;
PetscInt last_iter_count = 0;
PetscInt monitor_dump_frequency;
char monitor_output_dir[300];

// analysis params
analysisStruct myAnalysisStruct;


/* sets up monitoring of l1 runs (output of solution every few iterations, etc) */
#undef __FUNCT__
#define __FUNCT__ "setMonitorParams"
void setMonitorParams(PetscInt dump_frequency, char *output_dir){
    monitor_dump_frequency = dump_frequency;
    strcpy(monitor_output_dir, output_dir);
    monitor_save_number = 0;
}

#undef __FUNCT__
#define __FUNCT__ "setIterStartNumber"
void setIterStartNumber(PetscInt last_iter){
    last_iter_count = last_iter; 
}

/* set monitor state : on or off */
#undef __FUNCT__
#define __FUNCT__ "setMonitorState"
void setMonitorState(PetscInt flag){
    if(flag == 0){
        monitor_flag = 0;
    }
    else{
        monitor_flag = 1;
    }
}


/* monitors solution of l1 run */
#undef __FUNCT__
#define __FUNCT__ "monitorSolution"
PetscErrorCode monitorSolution(Vec xn, PetscInt iter_num, PetscScalar tau, char *method){
    PetscErrorCode ierr;
    PetscViewer pv;

    char *output_path, *tau_str, *iter_num_str, *save_num_str;
    ierr = PetscMalloc(500*sizeof(char),&output_path); CHKERRQ(ierr);
    ierr = PetscMalloc(50*sizeof(char),&tau_str); CHKERRQ(ierr);
    ierr = PetscMalloc(50*sizeof(char),&iter_num_str); CHKERRQ(ierr);
    ierr = PetscMalloc(50*sizeof(char),&save_num_str); CHKERRQ(ierr);


    // update iter_num in case of restart
    iter_num = iter_num + last_iter_count;

    // if applicable, record solution
    if( iter_num % monitor_dump_frequency == 0 ){
        // set up output path
        sprintf(tau_str,"%f",tau);
        sprintf(iter_num_str,"%lld",iter_num);
        sprintf(save_num_str,"%lld",monitor_save_number);

        strcpy(output_path, monitor_output_dir);
        strcat(output_path,"/");
        strcat(output_path,method);
        strcat(output_path,"_iter_");
        strcat(output_path,iter_num_str);
        //strcat(output_path,"_number_");
        //strcat(output_path,save_num_str);
        strcat(output_path,".petsc");

        // record solution to binary file
        PetscPrintf(PETSC_COMM_WORLD,"write monitor file to disk: %s ; iter_num = %d; last_iter_count = %d\n", output_path, iter_num, last_iter_count);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,output_path,FILE_MODE_WRITE,&pv);CHKERRQ(ierr);
        ierr = VecView(xn,pv); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(pv);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD,"done write monitor file to disk\n");

        monitor_save_number++;
    }

    // free strings
    ierr = PetscFree(output_path); CHKERRQ(ierr);
    ierr = PetscFree(tau_str); CHKERRQ(ierr);
    ierr = PetscFree(iter_num_str); CHKERRQ(ierr);
    ierr = PetscFree(save_num_str); CHKERRQ(ierr);

    return ierr;
}


/* set params for monitoring quantities like residuals, cost function, nnzs, etc */
#undef __FUNCT__
#define __FUNCT__ "setAnalysisOptions"
void setAnalysisOptions(PetscInt maxiters){
    PetscInt i;
    myAnalysisStruct.residuals = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    myAnalysisStruct.Js = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    myAnalysisStruct.num_nnzs = (PetscInt*)malloc(maxiters*sizeof(PetscInt));
    myAnalysisStruct.percent_errors = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    myAnalysisStruct.wn_norms = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    myAnalysisStruct.wn_norms_wavelet = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    myAnalysisStruct.xn_percent_diffs = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    myAnalysisStruct.wn_percent_diffs = (PetscScalar*)malloc(maxiters*sizeof(PetscScalar));
    for(i=0; i<maxiters; i++){
        myAnalysisStruct.residuals[i] = 0;
        myAnalysisStruct.Js[i] = 0;
        myAnalysisStruct.num_nnzs[i] = 0;
        myAnalysisStruct.percent_errors[i] = 0;
        myAnalysisStruct.wn_norms[i] = 0;
        myAnalysisStruct.wn_norms_wavelet[i] = 0;
        myAnalysisStruct.xn_percent_diffs[i] = 0;
        myAnalysisStruct.wn_percent_diffs[i] = 0;
    }
}


/* 
performs thresholded Landweber iteration:

x^{n+1} = St_tau[ x^{n} + A^{t}*b - A^{t}*A*x^{n} ]

where St_tau[x] = soft_thresholding[x] = for(j=1..n){ x[j] = (z[j]>tau) ? z[j]-tau : ((z[j]<-tau)? z[j]+tau :
0); }

set z = A^{t}*b (only calculated once)

set x^{0} = 0
for i=1:niter
    v^{n} = A*x^{n};
    w^{n} = A^{t}*A*x^{n} = A^{t}*v^{n}
    x^{n+1} = St[ x^{n} + z - w^{n} ];
end
*/
#undef __FUNCT__
#define __FUNCT__ "thresholdedLandweber"
PetscErrorCode thresholdedLandweber(Mat A, Vec b, Vec true_solution, PetscScalar tau, Vec x0, PetscScalar TOL, PetscInt maxiters, Vec * output, PetscInt * numiters, analysisStruct * analysis_out){
    Vec xn, vn, wn, dn, tn, z;
    PetscErrorCode ierr ;
    PetscInt       i,m,n,mloc,nloc,num_nnz; 
    PetscReal      rval, norm_val, norm_val1, norm_val2, un, un_prev;

    /* get mat sizes  - global and local */
    ierr  = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr  = MatGetLocalSize(A,&mloc,&nloc);CHKERRQ(ierr);

    /* calculate z = A^{t} * b */    
    VecCreate(PETSC_COMM_WORLD,&z);
    VecSetSizes(z,PETSC_DECIDE,n);
    VecSetFromOptions(z);
    //PetscPrintf(PETSC_COMM_WORLD,"Multiplying A^{t} by b and saving in z.\n");
    ierr = MatMultTranspose(A,b,z); CHKERRQ(ierr);

    /* check tau */
    VecMax(z,&i,&rval);
    if (tau > rval){
        PetscPrintf(PETSC_COMM_WORLD,"tau (%f) exceeds ||A^t b||_inf\n", tau);
    }

    /* set up vector xn and set it to the intial x0 */
    VecCreate(PETSC_COMM_WORLD,&xn);
    VecSetSizes(xn,PETSC_DECIDE,n);
    VecSetFromOptions(xn);
    ierr = VecCopy(x0,xn); CHKERRQ(ierr);


    /* now perform the Landweber iterations -------> */
    VecCreate(PETSC_COMM_WORLD,&vn);
    VecSetSizes(vn,PETSC_DECIDE,m);
    VecSetFromOptions(vn);
    VecCreate(PETSC_COMM_WORLD,&wn);
    VecSetSizes(wn,PETSC_DECIDE,n);
    VecSetFromOptions(wn);
    VecCreate(PETSC_COMM_WORLD,&tn);
    VecSetSizes(tn,PETSC_DECIDE,n);
    VecSetFromOptions(tn);
    VecCreate(PETSC_COMM_WORLD,&dn);
    VecSetSizes(dn,PETSC_DECIDE,m);
    VecSetFromOptions(dn);
    for(i=1; i<=maxiters; i++){
        PetscPrintf(PETSC_COMM_WORLD,"in iteration %d of %d\n", i, maxiters);
        if (i==1){
            // compute norms for x_{n}
            VecNorm(xn,NORM_1,&norm_val1); // ||x_{n}||_1
            ierr = MatMult(A,xn,vn); CHKERRQ(ierr); // vn = A*xn
            ierr = VecCopy(vn,dn); CHKERRQ(ierr); // dn = vn
            ierr = VecAXPY(dn,-1,b); CHKERRQ(ierr); // dn = -1*b + dn = -b + A*xn
            VecNorm(dn,NORM_2,&norm_val2); // ||A xn - b||_2
            un_prev = 2*tau*norm_val1 + norm_val2*norm_val2; // un = 2*tau*||xn|_1 + ||Axn - b||_2^2 

            // compute residuals and Js
            PetscPrintf(PETSC_COMM_WORLD,"computing norms1\n");
            myAnalysisStruct.residuals[i-1] = norm_val2;
            myAnalysisStruct.Js[i-1] = un_prev;
            num_nnz = 0;
            ierr = computeNumNNZs(xn, &num_nnz, 1e-15); CHKERRQ(ierr);
            myAnalysisStruct.num_nnzs[i-1] = num_nnz;
            ierr = VecCopy(xn,tn); CHKERRQ(ierr); // tn = xn
            ierr = VecAXPY(tn,-1,true_solution); CHKERRQ(ierr); // tn = -1*x + tn
            VecNorm(tn,NORM_2,&norm_val1);
            VecNorm(true_solution,NORM_2,&norm_val2);
            norm_val2 = 100*norm_val1/norm_val2; // 100*||xn - x||/||x||
            myAnalysisStruct.xn_percent_diffs[i-1] = norm_val2;
        }
        else{
            un_prev = un;
        }

        // vn = A*xn
        ierr = MatMult(A,xn,vn); CHKERRQ(ierr);

        //PetscPrintf(PETSC_COMM_WORLD,"wn = A^{t}*vn\n");
        ierr = MatMultTranspose(A,vn,wn); CHKERRQ(ierr);
        //PetscPrintf(PETSC_COMM_WORLD,"xn = z + xn\n");
        // xn = 1*z + xn
        ierr = VecAXPY(xn,1,z); 
        //PetscPrintf(PETSC_COMM_WORLD,"xn = xn - wn\n");
        // xn = -1*wn + xn
        ierr = VecAXPY(xn,-1,wn); 
        //PetscPrintf(PETSC_COMM_WORLD,"xn = Sl[wn]\n");
        // wn = xn
        //VecCopy(xn,wn);
        //ierr = softThreshold(wn, &xn, tau); CHKERRQ(ierr);
        ierr = VecScale(xn,1/(1+tau)); CHKERRQ(ierr);

        // compute norms for x_{n+1}
        VecNorm(xn,NORM_1,&norm_val); // ||x_{n+1}||_1
        ierr = MatMult(A,xn,vn); CHKERRQ(ierr); // vn = A*x_{n+1}
        ierr = VecCopy(vn,dn); CHKERRQ(ierr); // dn = vn
        ierr = VecAXPY(dn,-1,b); CHKERRQ(ierr); // dn = -1*b + dn = -b + A*x_{n+1}
        VecNorm(dn,NORM_2,&norm_val2); // ||A x_{n+1} - b||_2
        un = 2*tau*norm_val1 + norm_val2*norm_val2; // un1 = 2*tau*||x_{n+1}|_1 + ||Ax_{n+1} - b||_2^2 

        // compute residuals and Js
        PetscPrintf(PETSC_COMM_WORLD,"computing norms2\n");
        myAnalysisStruct.residuals[i] = norm_val2;
        myAnalysisStruct.Js[i] = un_prev;
        num_nnz = 0;
        ierr = computeNumNNZs(xn, &num_nnz, 1e-15); CHKERRQ(ierr);
        myAnalysisStruct.num_nnzs[i] = num_nnz;
        ierr = VecCopy(xn,tn); CHKERRQ(ierr); // tn = xn
        ierr = VecAXPY(tn,-1,true_solution); CHKERRQ(ierr); // tn = -1*x + tn
        VecNorm(tn,NORM_2,&norm_val1);
        VecNorm(true_solution,NORM_2,&norm_val2);
        norm_val2 = 100*norm_val1/norm_val2; // 100*||xn - x||/||x||
        myAnalysisStruct.xn_percent_diffs[i] = norm_val2;


        // check for convergence and exit if needed
        *numiters = i; 
        //PetscPrintf(PETSC_COMM_WORLD, "iteration %d: 100*fabs(un - un1)/un = %f\n", i, 100*fabs(un - un1)/un); 
        if( i>3 && (100*fabs(un - un_prev)/un < TOL) ){
            break; 
        }
    } 

    /* copy xn to output */ 
    VecCopy(xn, *output); 

    /* set analysis output */
    *analysis_out = myAnalysisStruct; 
    

    /* free memory */
    ierr = VecDestroy(z); CHKERRQ(ierr);
    ierr = VecDestroy(xn); CHKERRQ(ierr);
    ierr = VecDestroy(vn); CHKERRQ(ierr);
    ierr = VecDestroy(wn); CHKERRQ(ierr);
    ierr = VecDestroy(tn); CHKERRQ(ierr);
    ierr = VecDestroy(dn); CHKERRQ(ierr);

    return ierr;
}




/* 
performs thresholded FISTA iteration:

% let T(x) = S_tau( x + A'*b - A'*A*x )
% next,
% pick some x0 and iterate:
% x_{n+1}=T(x_n+\frac{t_{n}-1}{t_{n+1}} (x^{(n)}-x^{(n-1)}))
% where {t_n} is a sequence of numbers, generated by:
% t_{n+1}=\frac{1+\sqrt{1+4(t_{n})^2}}{2}$ and t_1=1

where S_tau[x] = soft_thresholding[x] = for(j=1..n){ x[j] = (z[j]>tau) ? z[j]-tau : ((z[j]<-tau)? z[j]+tau : 0); }
*/
#undef __FUNCT__
#define __FUNCT__ "thresholdedFista"
PetscErrorCode thresholdedFista(Mat A, Vec b, PetscScalar tau, Vec x0, PetscScalar TOL, PetscInt maxiters, Vec * output, PetscInt * numiters, PetscScalar * residuals, PetscScalar * Js)
{
    Vec pn, xn, xn_prev, tmp_vec, vn, wn, dn, z;
    PetscErrorCode ierr ;
    PetscInt       i,m,n,mloc,nloc; 
    PetscReal      rval, norm_val, norm_val1, norm_val2, un, un_prev;
    PetscScalar    *tns, tn_mult;

    /* get mat sizes  - global and local */
    ierr  = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr  = MatGetLocalSize(A,&mloc,&nloc);CHKERRQ(ierr);

    /* calculate z = A^{t} * b */    
    VecCreate(PETSC_COMM_WORLD,&z);
    VecSetSizes(z,PETSC_DECIDE,n);
    VecSetFromOptions(z);
    //PetscPrintf(PETSC_COMM_WORLD,"Multiplying A^{t} by b and saving in z.\n");
    ierr = MatMultTranspose(A,b,z); CHKERRQ(ierr);

    /* check tau */
    VecMax(z,&i,&rval);
    if (tau > rval){
        PetscPrintf(PETSC_COMM_WORLD,"tau (%f) exceeds ||A^t b||_inf\n", tau);
    }

    /* set up array of tns on each processor */
    ierr = PetscMalloc((maxiters+10)*sizeof(PetscScalar),&tns); CHKERRQ(ierr);
    tns[0] = 1;
    tns[1] = 1;
    for(i=2; i<(maxiters+10); i++){
        tns[i] = (1 + sqrt(1 + 4*tns[i-1]*tns[i-1]))/2;
    } 

    /* set up vectors xn, xn_prev, tmp_vec, and pn and set xn to the intial x0 */
    VecCreate(PETSC_COMM_WORLD,&xn);
    VecSetSizes(xn,PETSC_DECIDE,n);
    VecSetFromOptions(xn);
    VecCreate(PETSC_COMM_WORLD,&xn_prev);
    VecSetSizes(xn_prev,PETSC_DECIDE,n);
    VecSetFromOptions(xn_prev);
    VecCreate(PETSC_COMM_WORLD,&tmp_vec);
    VecSetSizes(tmp_vec,PETSC_DECIDE,n);
    VecSetFromOptions(tmp_vec);
    VecCreate(PETSC_COMM_WORLD,&pn);
    VecSetSizes(pn,PETSC_DECIDE,n);
    VecSetFromOptions(pn);
    ierr = VecCopy(x0,xn); CHKERRQ(ierr);


    /* now perform the Landweber iterations -------> */
    VecCreate(PETSC_COMM_WORLD,&vn);
    VecSetSizes(vn,PETSC_DECIDE,m);
    VecSetFromOptions(vn);
    VecCreate(PETSC_COMM_WORLD,&wn);
    VecSetSizes(wn,PETSC_DECIDE,n);
    VecSetFromOptions(wn);
    VecCreate(PETSC_COMM_WORLD,&dn);
    VecSetSizes(dn,PETSC_DECIDE,m);
    VecSetFromOptions(dn);
    for(i=1; i<=maxiters; i++){

        PetscPrintf(PETSC_COMM_WORLD,"in iteration %d of %d\n", i, maxiters);

        if (i==1){
            //set pn to xn
            VecCopy(xn,pn);
        
            // compute norms for x_{n}
            VecNorm(xn,NORM_1,&norm_val1); // ||x_{n}||_1
            ierr = MatMult(A,xn,vn); CHKERRQ(ierr); // vn = A*xn
            ierr = VecCopy(vn,dn); CHKERRQ(ierr); // dn = vn
            ierr = VecAXPY(dn,-1,b); CHKERRQ(ierr); // dn = -1*b + dn = -b + A*xn
            VecNorm(dn,NORM_2,&norm_val2); // ||A xn - b||_2
            un_prev = 2*tau*norm_val1 + norm_val2*norm_val2; // un = 2*tau*||xn|_1 + ||Axn - b||_2^2 

            // compute residuals and Js
            residuals[i-1] = norm_val2;
            Js[i-1] = un_prev;
        }
        else{
            //set pn to something else
            //pn = xn + ((tns(i)-1)/tns(i+1))*(xn - xn1);
            tn_mult = (tns[i]-1)/tns[i+1];
            ierr = VecCopy(xn,tmp_vec); CHKERRQ(ierr); // tmp_vec = xn
            // tmp_vec = -1*xn_prev + tmp_vec = -xn_prev+xn
            ierr = VecAXPY(tmp_vec,-1,xn_prev); CHKERRQ(ierr); 
            // pn = 1*xn + tn_mult*tmp_vec + 0*pn = xn + tn_mult*(xn-xn_prev)
            ierr = VecAXPBYPCZ(pn,1,tn_mult,0,xn,tmp_vec); CHKERRQ(ierr);

            //recycle norm calculation
            un_prev = un;
        }

        // x_{n-1} = x_n
        VecCopy(xn,xn_prev);

        // let x_{n+1} = T(pn); T(x) = S_tau( x + A'*b - A'*A*x )

        // vn = A*pn
        ierr = MatMult(A,pn,vn); CHKERRQ(ierr);

        // wn =  A^t vn = A^t A pn
        ierr = MatMultTranspose(A,vn,wn); CHKERRQ(ierr);
    
        // tmp_vec = pn
        VecCopy(pn,tmp_vec);

        // tmp_vec = 1*z + tmp_vec = z + pn = A^t*b + pn
        ierr = VecAXPY(tmp_vec,1,z); CHKERRQ(ierr); 

        // tmp_vec = -1*wn + tmp_vec = A^t*b + pn - wn = pn + A^t*b - A^t*A*pn
        ierr = VecAXPY(tmp_vec,-1,wn); CHKERRQ(ierr); 

        // xn = S_tau[tmp_vec]
        ierr = softThreshold(tmp_vec, &xn, tau); CHKERRQ(ierr);


        // compute norms for x_{n+1}
        VecNorm(xn,NORM_1,&norm_val); // ||x_{n+1}||_1
        ierr = MatMult(A,xn,vn); CHKERRQ(ierr); // vn = A*x_{n+1}
        ierr = VecCopy(vn,dn); CHKERRQ(ierr); // dn = vn
        ierr = VecAXPY(dn,-1,b); CHKERRQ(ierr); // dn = -1*b + dn = -b + A*x_{n+1}
        VecNorm(dn,NORM_2,&norm_val2); // ||A x_{n+1} - b||_2
        un = 2*tau*norm_val1 + norm_val2*norm_val2; // un1 = 2*tau*||x_{n+1}|_1 + ||Ax_{n+1} - b||_2^2 

        
        // compute residuals and Js
        residuals[i] = norm_val2;
        Js[i] = un;


        // check for convergence and exit if needed
        *numiters = i; 
        //PetscPrintf(PETSC_COMM_WORLD, "iteration %d: 100*fabs(un - un1)/un = %f\n", i, 100*fabs(un - un1)/un); 
        if( i>3 && (100*fabs(un - un_prev)/un < TOL) ){
            break; 
        }
    } 

    /* copy xn to output */ 
    VecCopy(xn, *output); 

    /* free memory */
    ierr = VecDestroy(z); CHKERRQ(ierr);
    ierr = VecDestroy(xn); CHKERRQ(ierr);
    ierr = VecDestroy(xn_prev); CHKERRQ(ierr);
    ierr = VecDestroy(dn); CHKERRQ(ierr);
    ierr = VecDestroy(vn); CHKERRQ(ierr);
    ierr = VecDestroy(wn); CHKERRQ(ierr);
    ierr = VecDestroy(tmp_vec); CHKERRQ(ierr);
    ierr = PetscFree(tns); CHKERRQ(ierr);

    return ierr;
}



/* compute number of nnzs in input vector; define nonzero as something 
whose absolute value is larger than the supplied tolerance */
#undef __FUNCT__
#define __FUNCT__ "computeNumNNZs"
PetscErrorCode computeNumNNZs(Vec input, PetscInt *num_nnz, PetscScalar TOL) {
    PetscInt        i, n, num_nnz_self, num_nnz_root, low, high, *ix, ni;
    PetscErrorCode  ierr;
    PetscScalar     *y,vval; 
    PetscMPIInt    rank, size;
    MPI_Status status;

    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    // get input size
    ierr = VecGetSize(input,&n); CHKERRQ(ierr);

    ierr = PetscMalloc(sizeof(PetscInt), &ix); CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar), &y); CHKERRQ(ierr);
    
    ierr = VecGetOwnershipRange(input,&low,&high); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(input); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(input); CHKERRQ(ierr);

    *num_nnz = 0;
    num_nnz_self = 0;

    for(i=low; i<high; i++){
        // get value of v at global index i
        ni = 1;
        ix[0] = i;
        VecGetValues(input,ni,ix,y);
        vval = y[0]; 

        if( fabs(vval) > TOL ){
            num_nnz_self++;
        }
    }


    // send num_nnz_self to root to get count
    if(rank == 0){
        // MPI_RECV
        for(i=1; i<size; i++){
            MPI_Recv(&num_nnz_root, 1, MPI_INT, i, 103, MPI_COMM_WORLD, &status); 
            *num_nnz += num_nnz_root;
        }

        // add contribution from root
        *num_nnz += num_nnz_self;
    }
    else{
        // MPI_SEND
        MPI_Send(&num_nnz_self, 1, MPI_INT, 0, 103, MPI_COMM_WORLD);
    }


    // free memory
    ierr = PetscFree(ix); CHKERRQ(ierr);
    ierr = PetscFree(y); CHKERRQ(ierr);

    return ierr;
}




#undef __FUNCT__
#define __FUNCT__ "softThreshold"
PetscErrorCode softThreshold(Vec input, Vec *output, PetscScalar tau) {

    Vec output_local;
    PetscInt        i, n, low, high, *ix, ni;
    PetscErrorCode  ierr;
    PetscScalar     *y,vval; 

    // get input size
    ierr = VecGetSize(input,&n); CHKERRQ(ierr);

    // setup output_local vector (don't creat on *output or you will leak memory)
    ierr = VecCreate(PETSC_COMM_WORLD,&output_local); CHKERRQ(ierr);
    ierr = VecSetSizes(output_local,PETSC_DECIDE,n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(output_local); CHKERRQ(ierr);
    ierr = VecCopy(input,output_local); CHKERRQ(ierr);

    ierr = PetscMalloc(sizeof(PetscInt), &ix); CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar), &y); CHKERRQ(ierr);
    
    ierr = VecGetOwnershipRange(input,&low,&high); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(input); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(input); CHKERRQ(ierr);

    for(i=low; i<high; i++){
        
        // get value of v at global index i
        ni = 1;
        ix[0] = i;
        VecGetValues(input,ni,ix,y);
        vval = y[0]; 

        if( vval > tau ){
            ierr = VecSetValue(output_local, i, vval - tau, INSERT_VALUES); CHKERRQ(ierr);
        }
        else if( vval < -tau ){
            ierr = VecSetValue(output_local, i, vval + tau, INSERT_VALUES); CHKERRQ(ierr);
        }
        else{
            ierr = VecSetValue(output_local, i, 0, INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecAssemblyBegin(output_local); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(output_local); CHKERRQ(ierr);

    ierr = VecCopy(output_local,*output); CHKERRQ(ierr);

    // free memory
    ierr = VecDestroy(output_local); CHKERRQ(ierr);
    ierr = PetscFree(ix); CHKERRQ(ierr);
    ierr = PetscFree(y); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "hardThreshold"
PetscErrorCode hardThreshold(Vec input, Vec *output, PetscScalar tau) {

    Vec output_local;
    PetscInt        i, n, low, high, *ix, ni;
    PetscErrorCode  ierr;
    PetscScalar     *y,vval; 

    // get input size
    ierr = VecGetSize(input,&n); CHKERRQ(ierr);

    // setup output_local vector (don't creat on *output or you will leak memory)
    ierr = VecCreate(PETSC_COMM_WORLD,&output_local); CHKERRQ(ierr);
    ierr = VecSetSizes(output_local,PETSC_DECIDE,n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(output_local); CHKERRQ(ierr);
    ierr = VecCopy(input,output_local); CHKERRQ(ierr);

    ierr = PetscMalloc(sizeof(PetscInt), &ix); CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar), &y); CHKERRQ(ierr);
    
    ierr = VecGetOwnershipRange(input,&low,&high); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(input); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(input); CHKERRQ(ierr);

    for(i=low; i<high; i++){
        
        // get value of v at global index i
        ni = 1;
        ix[0] = i;
        VecGetValues(input,ni,ix,y);
        vval = y[0]; 

        if( vval > tau ){
            ierr = VecSetValue(output_local, i, vval, INSERT_VALUES); CHKERRQ(ierr);
        }
        else if( vval < -tau ){
            ierr = VecSetValue(output_local, i, vval, INSERT_VALUES); CHKERRQ(ierr);
        }
        else{
            ierr = VecSetValue(output_local, i, 0, INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecAssemblyBegin(output_local); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(output_local); CHKERRQ(ierr);

    ierr = VecCopy(output_local,*output); CHKERRQ(ierr);

    // free memory
    ierr = VecDestroy(output_local); CHKERRQ(ierr);
    ierr = PetscFree(ix); CHKERRQ(ierr);
    ierr = PetscFree(y); CHKERRQ(ierr);

    return ierr;
}



/* a different type of soft thresholding:
S_theta,tau(x) = { x if x >= theta*tau; theta/(theta-1)*(x - tau) if tau < x < theta*tau ; 0 if
|x|<=tau; theta/(theta-1)*(x + tau) if -theta*tau < x < -tau; x if x<= -theta*tau }
*/
#undef __FUNCT__
#define __FUNCT__ "softThreshold2"
PetscErrorCode softThreshold2(Vec input, Vec *output, PetscScalar theta, PetscScalar tau) {

    Vec output_local;
    PetscInt        i, n, low, high, *ix, ni;
    PetscErrorCode  ierr;
    PetscScalar     *y,vval, thetatau; 

    thetatau = theta*tau;

    // get input size
    ierr = VecGetSize(input,&n); CHKERRQ(ierr);

    // setup output_local vector (don't creat on *output or you will leak memory)
    ierr = VecCreate(PETSC_COMM_WORLD,&output_local); CHKERRQ(ierr);
    ierr = VecSetSizes(output_local,PETSC_DECIDE,n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(output_local); CHKERRQ(ierr);
    ierr = VecCopy(input,output_local); CHKERRQ(ierr);

    ierr = PetscMalloc(sizeof(PetscInt), &ix); CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscScalar), &y); CHKERRQ(ierr);
    
    ierr = VecGetOwnershipRange(input,&low,&high); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(input); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(input); CHKERRQ(ierr);

    for(i=low; i<high; i++){
        
        // get value of v at global index i
        ni = 1;
        ix[0] = i;
        VecGetValues(input,ni,ix,y);
        vval = y[0]; 

        if( vval >= thetatau ){
            ierr = VecSetValue(output_local, i, vval, INSERT_VALUES); CHKERRQ(ierr);
        }
        else if( (vval > tau) && (vval < thetatau) ){
            ierr = VecSetValue(output_local, i, (PetscScalar)theta/(theta-1) * (vval - tau), INSERT_VALUES); CHKERRQ(ierr);
        }
        else if( (vval > -thetatau) && (vval < tau) ){
            ierr = VecSetValue(output_local, i, (PetscScalar)theta/(theta-1) * (vval - tau), INSERT_VALUES); CHKERRQ(ierr);
        }
        else if( fabs(vval) <= tau){
            ierr = VecSetValue(output_local, i, 0, INSERT_VALUES); CHKERRQ(ierr);
        }
        else if( vval <= -thetatau ){
            ierr = VecSetValue(output_local, i, vval, INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecAssemblyBegin(output_local); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(output_local); CHKERRQ(ierr);

    ierr = VecCopy(output_local,*output); CHKERRQ(ierr);

    // free memory
    ierr = VecDestroy(output_local); CHKERRQ(ierr);
    ierr = PetscFree(ix); CHKERRQ(ierr);
    ierr = PetscFree(y); CHKERRQ(ierr);

    return ierr;
}



#undef __FUNCT__
#define __FUNCT__ "projectOnL1Ball"
/* l2 projection of input onto l1 ball of radius R */
PetscErrorCode projectOnL1Ball(Vec input, Vec *output, PetscScalar R){

    Vec svec;
    PetscErrorCode ierr ;
    PetscInt i,j,k,k1,k2,n,ind,ni,myni,vecsize,mysize,return_code,low,high,mylow,myhigh,ind_in,ind_out,quit_loop, left, right, mid; 
    PetscInt       *ix;
    PetscScalar    val,tval;
    PetscReal      nval,nval1,nval2;
    PetscMPIInt    rank, size;
    PetscScalar *input_vec_vals, *input_vec_vals_sorted, *output_vec_vals;
    PetscScalar *local_vals,*my_local_vals;
    MPI_Status status;
    Node *input_node, *output_node;

    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    // get input size
    ierr = VecGetSize(input,&vecsize); CHKERRQ(ierr);

    // setup output vector
    //ierr = VecCreate(PETSC_COMM_WORLD,output); CHKERRQ(ierr);
    //ierr = VecSetSizes(*output,PETSC_DECIDE,vecsize); CHKERRQ(ierr);
    //ierr = VecSetFromOptions(*output); CHKERRQ(ierr);

    // setup vector to hold soft-thresholding result
    ierr = VecCreate(PETSC_COMM_WORLD,&svec); CHKERRQ(ierr);
    ierr = VecSetSizes(svec,PETSC_DECIDE,vecsize); CHKERRQ(ierr);
    ierr = VecSetFromOptions(svec); CHKERRQ(ierr);

    // check different cases of R
    ierr = VecNorm(input,NORM_1,&nval); CHKERRQ(ierr);  
    if(R == 0){
        PetscPrintf(PETSC_COMM_WORLD,"------> R=0!!!\n");
        ierr = VecSet(*output,0.0); CHKERRQ(ierr);       
        return ierr;
    }
    else if(nval <= R){
        PetscPrintf(PETSC_COMM_WORLD,"------> nval <= R!!!\n");
        ierr = VecCopy(input,*output); CHKERRQ(ierr);
        return ierr;
    }

    // otherwise, nval > R and we must do interpolation
    // get all the vec values on the root processor --->

    // get ranges for this process
    ierr = VecGetOwnershipRange(input,&mylow,&myhigh); CHKERRQ(ierr);

    // get these values
    ierr = VecAssemblyBegin(input); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(input); CHKERRQ(ierr);
    myni = myhigh - mylow;
    ix = (PetscInt*)malloc(myni*sizeof(PetscInt));
    my_local_vals = (PetscScalar*)malloc(myni*sizeof(PetscScalar));
    for(i=0; i<myni; i++){
        ix[i] = mylow + i;
    }
    ierr = VecGetValues(input,myni,ix,my_local_vals); CHKERRQ(ierr);

    // make space before send/receive
    if( rank == 0 ){
        input_vec_vals = (PetscScalar*)malloc(vecsize*sizeof(PetscScalar)); 

        // stick in the values from root itself
        for(j=0; j<myni; j++){
            ind = mylow + j;
            input_vec_vals[ind] = my_local_vals[j]; 
        }
    }
    
    // if root, receive
    if( rank == 0 ){
        for(i=1; i<size; i++){
            // receive low and high 
            MPI_Recv(&low, 1, MPI_LONG_LONG_INT, i, 101, MPI_COMM_WORLD, &status);
            MPI_Recv(&high, 1, MPI_LONG_LONG_INT, i, 102, MPI_COMM_WORLD, &status);

            // recieve vec vals from processors
            ni = high - low;
            local_vals = (PetscScalar*)malloc(ni*sizeof(PetscScalar));
            MPI_Recv(local_vals, ni, MPI_DOUBLE, i, 103, MPI_COMM_WORLD, &status); 

            // stick the values in the right places of the 1d vector
            for(j=0; j<ni; j++){
                ind = low + j;
                input_vec_vals[ind] = local_vals[j]; 
            } 

            // free stuff
            free(local_vals);
        }
    }
    else{
        // send local_vals to root
        MPI_Send(&mylow, 1, MPI_LONG_LONG_INT, 0, 101, MPI_COMM_WORLD);
        MPI_Send(&myhigh, 1, MPI_LONG_LONG_INT, 0, 102, MPI_COMM_WORLD);
        MPI_Send(my_local_vals, myni, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD);
    }

    
    // now we have local vals on root processor, so sort them
    input_vec_vals_sorted = (PetscScalar*)malloc((vecsize+1)*sizeof(PetscScalar));
    if(rank == 0){
        for(i=0; i<vecsize; i++){
            input_vec_vals_sorted[i] = fabs(input_vec_vals[i]);
        }
        // append zero to end
        input_vec_vals_sorted[vecsize] = 0;
        
        // Sort the series X using ANSI C qsort() function
        // we sort in descending order with the corresponding compfcn
        qsort(input_vec_vals_sorted, vecsize, sizeof(double), compfcn);    
    }
    
    // first, broadcast input_vec_vals_sorted to all processors
    MPI_Bcast(input_vec_vals_sorted,vecsize+1,MPI_DOUBLE,0,MPI_COMM_WORLD);

    // variable to control if we are done calculating k1,k2
    quit_loop = 0;

    // first, test end points
    k = 0; 
    ierr = softThreshold(input, &svec, input_vec_vals_sorted[k]); CHKERRQ(ierr);
    ierr = VecNorm(svec,NORM_1,&nval1); CHKERRQ(ierr);
    ierr = softThreshold(input, &svec, input_vec_vals_sorted[k+1]); CHKERRQ(ierr);
    ierr = VecNorm(svec,NORM_1,&nval2); CHKERRQ(ierr);
    if( (nval1 <= R) && (nval2 > R) ){
	k1 = k;
	k2 = k+1;
	quit_loop = 1;
    }

    k = vecsize - 1; 
    ierr = softThreshold(input, &svec, input_vec_vals_sorted[k]); CHKERRQ(ierr);
    ierr = VecNorm(svec,NORM_1,&nval1); CHKERRQ(ierr);
    ierr = softThreshold(input, &svec, input_vec_vals_sorted[k+1]); CHKERRQ(ierr);
    ierr = VecNorm(svec,NORM_1,&nval2); CHKERRQ(ierr);
    if( (nval1 <= R) && (nval2 > R) ){
	k1 = k;
	k2 = k+1;
	quit_loop = 1;
    }

    // perform bisection --->
    left = 0;
    right = vecsize; 
    
    while( (abs(right - left) > 1) && (quit_loop != 1) ){
        k1 = left;
	k2 = right;
	mid = round((left + right)/2);
	
        ierr = softThreshold(input, &svec, input_vec_vals_sorted[mid]); CHKERRQ(ierr);
        ierr = VecNorm(svec,NORM_1,&nval); CHKERRQ(ierr);
    
        if( nval <= R ){
            left = mid;
            ierr = softThreshold(input, &svec, input_vec_vals_sorted[left]); CHKERRQ(ierr);
            ierr = VecNorm(svec,NORM_1,&nval1); CHKERRQ(ierr);
            ierr = softThreshold(input, &svec, input_vec_vals_sorted[left+1]); CHKERRQ(ierr);
            ierr = VecNorm(svec,NORM_1,&nval2); CHKERRQ(ierr);
            if((nval1 <= R) && (nval2 > R)){
                k1 = left;
                k2 = left+1;
                quit_loop = 1;
            }
        }

        else{
            right = mid;
            ierr = softThreshold(input, &svec, input_vec_vals_sorted[right]); CHKERRQ(ierr);
            ierr = VecNorm(svec,NORM_1,&nval1); CHKERRQ(ierr);
            ierr = softThreshold(input, &svec, input_vec_vals_sorted[right+1]); CHKERRQ(ierr);
            ierr = VecNorm(svec,NORM_1,&nval2); CHKERRQ(ierr);
            if((nval1 <= R) && (nval2 > R)){
                k1 = right;
                k2 = right+1;
                quit_loop = 1;
            }
        }
    }

    /* now that we have indices, do linear interpolation to find k
    % (sa(k1), val1), (sa(k2), val2) ==> (k,R) for sa(k1)<=k<=sa(k2)
    % (y - y0)/(y1 - y0) = (x - x0)/(x1 - x0)
    % => x = ((y - y0)/(y1 - y0))*(x1 - x0) + x0
    % => k = ((R - val1)/(val2 - val1))*(sa(k2) - sa(k1)) + sa(k1) */
    ierr = softThreshold(input, &svec, input_vec_vals_sorted[k1]); CHKERRQ(ierr);
    ierr = VecNorm(svec,NORM_1,&nval1); CHKERRQ(ierr);
    ierr = softThreshold(input, &svec, input_vec_vals_sorted[k2]); CHKERRQ(ierr);
    ierr = VecNorm(svec,NORM_1,&nval2); CHKERRQ(ierr);

    if(nval1 == R){
        tval = input_vec_vals_sorted[k1];
    }
    else{
        tval = ((R - nval1)/(nval2 - nval1))*(input_vec_vals_sorted[k2] - input_vec_vals_sorted[k1]) + input_vec_vals_sorted[k1];
    }

    // now that we have the right k, perform soft thresholding
    // on the original data to get the projection
    ierr = softThreshold(input,&svec,tval); CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_WORLD,"k1 = %d; k2 = %d\n", k1,k2);
    //PetscPrintf(PETSC_COMM_WORLD,"tval = %f\n", tval);

    // copy result to output and exit
    ierr = VecCopy(svec,*output); CHKERRQ(ierr);

    // free stuff
    ierr = VecDestroy(svec); CHKERRQ(ierr);
    free(input_vec_vals_sorted);
    free(my_local_vals);
    free(ix);
    if( rank == 0 ){
        free(input_vec_vals);
    }

    return ierr;
}





/* 
soft thresholded coordinate descent
*/
#undef __FUNCT__
#define __FUNCT__ "thresholded_coordinate_descent"
PetscErrorCode thresholded_coordinate_descent(Mat A, Vec b, Vec true_solution, PetscScalar tau, Vec x0, PetscScalar TOL, PetscInt maxiters, Vec * output, PetscInt * numiters, analysisStruct * analysis_out){
    Vec xn, y, aj, vn, dn, tn;
    PetscErrorCode ierr ;
    PetscInt       i,j,ind,m,n,mloc,nloc,numiters_guess,maxiters_guess,num_nnz,low,high; 
    PetscInt       *ix;
    PetscReal      rval, norm_val, norm_val1, norm_val2, un, un_prev;
    PetscReal      *column_norms;
    PetscScalar    val1, val2, sum_val, aij, yi, bi, xj, Bj;
    PetscScalar    *ys, *residuals, *Js;

    /* get mat sizes  - global and local */
    ierr  = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    ierr  = MatGetLocalSize(A,&mloc,&nloc);CHKERRQ(ierr);

    /* setup arrays and vectors */    
    PetscMalloc(sizeof(PetscInt),&ix);
    PetscMalloc(sizeof(PetscScalar),&ys);

    VecCreate(PETSC_COMM_WORLD,&xn);
    VecSetSizes(xn,PETSC_DECIDE,n);
    VecSetFromOptions(xn);

    VecCreate(PETSC_COMM_WORLD,&y);
    VecSetSizes(y,PETSC_DECIDE,m);
    VecSetFromOptions(y);

    VecCreate(PETSC_COMM_WORLD,&aj);
    VecSetSizes(aj,PETSC_DECIDE,m);
    VecSetFromOptions(aj);

    VecCreate(PETSC_COMM_WORLD,&vn);
    VecSetSizes(vn,PETSC_DECIDE,m);
    VecSetFromOptions(vn);

    VecCreate(PETSC_COMM_WORLD,&dn);
    VecSetSizes(dn,PETSC_DECIDE,m);
    VecSetFromOptions(dn);

    VecCreate(PETSC_COMM_WORLD,&tn);
    VecSetSizes(tn,PETSC_DECIDE,n);
    VecSetFromOptions(tn);


    /* compute column norms of matrix */
    PetscPrintf(PETSC_COMM_WORLD, "computing column norms..\n");
    PetscMalloc(n*sizeof(PetscReal),&column_norms);
    ierr = MatGetColumnNorms(A,NORM_2,column_norms); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "done computing column norms\n");

    /* run FISTA to obtain initial guess, max 100 iterations */
    //maxiters_guess = 100;
    //PetscPrintf(PETSC_COMM_WORLD, "running FISTA pre-step for %d iterations\n", maxiters_guess);
    //ierr = thresholdedFista(A, b, tau, x0, TOL, maxiters_guess, &xn, &numiters_guess,residuals,Js); CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD, "done with FISTA\n");
    PetscPrintf(PETSC_COMM_WORLD, "in VecSet\n");
    ierr = VecSet(x0,0.0); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "in VecCopy\n");
    ierr = VecCopy(x0,xn); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "after VecCopy\n");

    /* now run coordinate descent */
    for(i=1; i<=maxiters; i++){

        PetscPrintf(PETSC_COMM_WORLD, "in iteration %d\n", i);

        /* compute various quantities ---> */
        ierr = VecNorm(xn,NORM_1,&norm_val1); CHKERRQ(ierr); // ||x_{n}||_1
        PetscPrintf(PETSC_COMM_WORLD,"done with norm calculation\n");
        ierr = MatMult(A,xn,vn); CHKERRQ(ierr); // vn = A*xn (mxn * nx1 = mx1)
        ierr = VecCopy(vn,dn); CHKERRQ(ierr); // dn = vn
        ierr = VecAXPY(dn,-1,b); CHKERRQ(ierr); // dn = -1*b + dn = -b + A*xn
        VecNorm(dn,NORM_2,&norm_val2); // ||A xn - b||_2
        un = 2*tau*norm_val1 + norm_val2*norm_val2; // un = 2*tau*||xn|_1 + ||Axn - b||_2^2 

        // compute residuals, Js, and num_nnzs
        PetscPrintf(PETSC_COMM_WORLD,"computing norms1\n");
        myAnalysisStruct.residuals[i-1] = norm_val2;
        myAnalysisStruct.Js[i-1] = un;
        num_nnz = 0;
        ierr = computeNumNNZs(xn, &num_nnz, 1e-15); CHKERRQ(ierr);
        myAnalysisStruct.num_nnzs[i-1] = num_nnz;
        ierr = VecCopy(xn,tn); CHKERRQ(ierr); // tn = xn
        ierr = VecAXPY(tn,-1,true_solution); CHKERRQ(ierr); // tn = -1*x + tn
        VecNorm(tn,NORM_2,&norm_val1);
        VecNorm(true_solution,NORM_2,&norm_val2);
        norm_val2 = 100*norm_val1/norm_val2; // 100*||xn - x||/||x||
        myAnalysisStruct.xn_percent_diffs[i-1] = norm_val2;


        /* pick index j to correct at random */
        j = round(( (PetscScalar)rand() / ((PetscScalar)(RAND_MAX)+(PetscScalar)(1)) )*(n-1)); // in [0,n-1)

        
        PetscPrintf(PETSC_COMM_WORLD,"in iteration %d of %d (with j = %d of %d)\n", i, maxiters, j, n-1);

        /* compute B_j = sum i=1..m { a_ij ( b_i - sum k=1..n,not j { a_ik x_k } } ----> */ 

        /* first, calculate y = A*xn */
        ierr = MatMult(A,xn,y); CHKERRQ(ierr);

        /* get j-th column vector */
        ierr = MatGetColumnVector(A,aj,j); CHKERRQ(ierr);

        /* get xj = xn(j) */
        //ierr = VecAssemblyBegin(xn); CHKERRQ(ierr);
        //ierr = VecAssemblyEnd(xn); CHKERRQ(ierr);
        ierr= VecGetOwnershipRange(xn,&low,&high); CHKERRQ(ierr);
        if( j>=low && j<high ){
            ix[0] = j;
            ierr = VecGetValues(xn,1,ix,ys); CHKERRQ(ierr);
            xj = ys[0];
        }

        PetscPrintf(PETSC_COMM_WORLD,"starting ind loop..\n");

        Bj = 0;
        for(ind = 0; ind<m; ind++){
           
            if(ind%1000 == 0) 
                PetscPrintf(PETSC_COMM_WORLD,"ind loop %d of %d\n", ind, m);

            /* get y[i] = (A*xn)[i] */
            ierr= VecGetOwnershipRange(y,&low,&high); CHKERRQ(ierr);
            if( ind>=low && ind<high ){
                ix[0] = ind;
                //ierr = VecAssemblyBegin(y); CHKERRQ(ierr);
                //ierr = VecAssemblyEnd(y); CHKERRQ(ierr);
                ierr = VecGetValues(y,1,ix,ys); CHKERRQ(ierr);
                yi = ys[0];
            }

            /* get a_ij from aj */
            ierr= VecGetOwnershipRange(aj,&low,&high); CHKERRQ(ierr);
            if( ind>=low && ind<high ){
                ix[0] = ind;
                //ierr = VecAssemblyBegin(aj); CHKERRQ(ierr);
                //ierr = VecAssemblyEnd(aj); CHKERRQ(ierr);
                ierr = VecGetValues(aj,1,ix,ys); CHKERRQ(ierr);
                aij = ys[0];
            }

            /* get bi from rhs b */
            ierr= VecGetOwnershipRange(b,&low,&high); CHKERRQ(ierr);
            if( ind>=low && ind<high ){
                ix[0] = ind;
                ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
                ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
                ierr = VecGetValues(b,1,ix,ys); CHKERRQ(ierr);
                bi = ys[0];
            }

            /* get sum k=1..n,not j [ a_ik x_k ]*/
            sum_val = yi - aij*xj;

            /* compute bi - sum_val */
            Bj += aij*(bi - sum_val);
        }

        /* soft-threshold Bj at tau */
        if(Bj > tau){
            xj = Bj - tau;
        }
        else if(Bj < -tau){
            xj = Bj + tau;
        }
        else{
            xj = 0;
        }
            

        PetscPrintf(PETSC_COMM_WORLD,"done with ind loop, updating xn..\n");
        /* update j-th component of xn */
        ierr= VecGetOwnershipRange(xn,&low,&high); CHKERRQ(ierr);
        if( j>=low && j<high ){
            if( column_norms[j] > 1e-8 ){
                ierr = VecSetValue(xn,j,xj/(column_norms[j]*column_norms[j]), INSERT_VALUES); CHKERRQ(ierr);
            }
            else{
                ierr = VecSetValue(xn,j,0, INSERT_VALUES); CHKERRQ(ierr);
            }
            ierr = VecAssemblyBegin(xn); CHKERRQ(ierr);
            ierr = VecAssemblyEnd(xn); CHKERRQ(ierr);
        }
    }


    /* copy xn to output */ 
    VecCopy(xn, *output); 

    /* free memory */
    ierr = VecDestroy(y); CHKERRQ(ierr);
    ierr = VecDestroy(aj); CHKERRQ(ierr);
    ierr = VecDestroy(vn); CHKERRQ(ierr);
    ierr = VecDestroy(dn); CHKERRQ(ierr);
    ierr = PetscFree(ix); CHKERRQ(ierr);
    ierr = PetscFree(ys); CHKERRQ(ierr);
    ierr = PetscFree(column_norms); CHKERRQ(ierr);

    return ierr;
}



#undef __FUNCT__
#define __FUNCT__ "getLargestSingularValue"
/* uses power iteration to find largest singular value;
see http://www-math.mit.edu/~persson/18.335/lec15handout6pp.pdf for details */
PetscErrorCode getLargestSingularValue(Mat A, PetscInt niters, PetscScalar *sval){
    Vec v,w,b;
    PetscScalar val;
    PetscReal norm_val;
    PetscInt i,m,n,p;
    PetscErrorCode ierr;

    /* get mat sizes */
    ierr  = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    if(m > n)
        p = n;
    else
        p = m;

    /* setup vectors */
    ierr = VecCreate(PETSC_COMM_WORLD,&v); CHKERRQ(ierr);
    ierr = VecSetSizes(v,PETSC_DECIDE,p); CHKERRQ(ierr);
    ierr = VecSetFromOptions(v); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&w); CHKERRQ(ierr);
    if(m>n){
        ierr = VecSetSizes(w,PETSC_DECIDE,m); CHKERRQ(ierr);
    }
    else{
        ierr = VecSetSizes(w,PETSC_DECIDE,n); CHKERRQ(ierr);
    }
    ierr = VecSetFromOptions(w); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
    ierr = VecSetSizes(b,PETSC_DECIDE,p); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
   
    /* init random unit vector */
    for(i=0; i<p; i++){
        val = ( (PetscScalar)rand() / ((PetscScalar)(RAND_MAX)+(PetscScalar)(1)) )*9; // in [0,9)
        ierr = VecSetValue(v,i,val,INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecAssemblyBegin(v); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(v); CHKERRQ(ierr);
        ierr = VecNorm(v,NORM_2,&norm_val); CHKERRQ(ierr);  
        ierr = VecScale(v,1/norm_val); //v = v/||v||
    }

    /* do power iteration */
    for(i=0; i<niters; i++){
        if(m > n){
            //b = A'*A*v
            ierr = MatMult(A,v,w); CHKERRQ(ierr); //w = A*v
            ierr = MatMultTranspose(A,w,b); CHKERRQ(ierr); //b=A'*w = A'*A*v
        }       
        else{
            //b = A*A'*v
            ierr = MatMultTranspose(A,v,w); CHKERRQ(ierr); //w = A'*v
            ierr = MatMult(A,w,b); CHKERRQ(ierr); //b = A*w = A*A'*v
        }

        //v = b/||b||
        ierr = VecNorm(b,NORM_2,&norm_val); CHKERRQ(ierr);
        ierr = VecScale(b,1/norm_val);
        ierr = VecCopy(b,v);

        //update singular value (=eigenvalue of A'*A or A*A')
        if(m > n){
            ierr = MatMult(A,v,w); CHKERRQ(ierr); // w = A*v
            ierr = MatMultTranspose(A,w,b); CHKERRQ(ierr); // b = A'*w = A'*A*v
        }
        else{
            ierr = MatMultTranspose(A,v,w); CHKERRQ(ierr); // w = A'*v
            ierr = MatMult(A,w,b); CHKERRQ(ierr); // b = A*w = A*A'*v
        }

        
        VecDot(v,b,sval);
        *sval = sqrt(*sval);
    }

    // free data
    ierr = VecDestroy(v); CHKERRQ(ierr);
    ierr = VecDestroy(w); CHKERRQ(ierr);
    ierr = VecDestroy(b); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "compfcn"
/* comparison function for C quicksort from stdlib in *descending order* */
int compfcn(const void *p, const void *q){
    PetscScalar *ptr1 = (PetscScalar *)(p);
    PetscScalar *ptr2 = (PetscScalar *)(q);

    /*if (*ptr1 < *ptr2)
    return -1;
    else if (*ptr1 == *ptr2)
    return 0;
    else
    return 1;*/

    if (*ptr1 < *ptr2)
        return 1;
    else if (*ptr1 == *ptr2)
        return 0;
    else
        return -1;
}


