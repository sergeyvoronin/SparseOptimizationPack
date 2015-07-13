#include <cusp/io/matrix_market.h>
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/print.h>

#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/blas.h>
#include <cusp/linear_operator.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


/* run some sparse recovery algorithms at particular tau 
Sergey Voronin */
namespace blas = cusp::blas;


/* compute soft threshold of vector */
__global__ void cudasoftThreshold(int N, float tau, const float* x, float * result){
   int i;
   int i0 = blockIdx.x*blockDim.x + threadIdx.x;
   float val;
   for(i=i0; i<N; i+=blockDim.x*gridDim.x){
      val = x[i];
      if(val > tau){
         result[i] = val - tau;
      }
      else if(val < tau &&  val > -tau){
         result[i] = 0;   
      }
      else{
         result[i] = val + tau;
      }
   }
}


/* compute p threshold of vector */
__global__ void cudapThreshold( int N, float tau, float p, const float* x, float * result) {
   int i,sgnval;
   int i0 = blockIdx.x*blockDim.x + threadIdx.x;
   float val, val2, TOL = 1e-15;
   
   // sign(x(i))*max(0,abs(x(i)) - tau*abs(x(i))^(p-1));
   for(i=i0;i<N;i+=blockDim.x*gridDim.x) {
      val = x[i];
      if(abs(val) < TOL){
         sgnval = 0;
      } else{
          if(val>0){ 
             sgnval = 1; 
          }
          else{ 
             sgnval = -1;
          }
      }

      val2 = abs(val) - tau*pow(abs(val), p-1);
      if(val2 > 0){
        result[i] = sgnval*val2;
      }
      else{
        result[i] = 0;
      }
   }
}


/* multiply a vector yn by a diagonal matrix D with diagonal elements diag_elems */
__global__ void cudaApplyD( int N, int *diag_elems, float* yn, float * result) {
   int i;
   int i0 = blockIdx.x*blockDim.x + threadIdx.x;
   float val;
   for(i=i0; i<N; i+=blockDim.x*gridDim.x){
        if( diag_elems[i] > 0 ){
            result[i] = yn[i];
        }else{
            result[i] = 0;
        }
   }
}


template <class LinearOperator, class Vector>
void fistaAlgorithm(LinearOperator &A, LinearOperator &At, Vector &x0, Vector &b, Vector &z, int num_iters, float tau, Vector &xn){
    int i, nthreads, nblocks, iter_num, m, n;
    float tn_mult;
    float *tns;
    const float *tvec_ptr;
    float *tvec_ptr2;

    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;
    
    // get dimensions
    m = A.num_rows;
    n = A.num_cols; 

    // setup tns
    tns = (float*)malloc((num_iters+10)*sizeof(float));
    tns[0] = 1;
    for(i=1; i<(num_iters+10); i++){
        tns[i] = (1 + sqrt(1 + 4*tns[i-1]*tns[i-1]))/2;
    }

    // setup x0, xn, xn1, pn
    cusp::array1d<ValueType, MemorySpace> xn1(n);
    cusp::array1d<ValueType, MemorySpace> pn(n);
    cusp::array1d<ValueType, MemorySpace> dn(n);
    cusp::array1d<ValueType, MemorySpace> vn(m);
    cusp::array1d<ValueType, MemorySpace> wn(n);
    cusp::array1d<ValueType, MemorySpace> threshold_vec(n);

    // initialize x0 to be zero vector
    //blas::scal(x0,0.0);
    xn = x0;
    xn1 = x0;

    /* perform iterations ---------> */
    for(iter_num=0; iter_num<num_iters; iter_num++){

        printf("doing iteration %d of %d\n", iter_num+1, num_iters);

        // setup pn array
        if(iter_num == 0){
            pn = xn;
        }
        else{
            //pn = xn + ((tns(i)-1)/tns(i+1))*(xn - xn1);
            
            //dn = xn - xn1
            dn = xn;
            blas::axpy( xn1, dn, -1.0f);

            tn_mult = (tns[iter_num]-1)/tns[iter_num+1];
            //pn = xn + tn_mult*dn
            pn = xn;
            blas::axpy( dn, pn, 1.0f*tn_mult);
        }

        // record x_{n-1}
        xn1 = xn;

        //vn = A*pn; (mxn * nx1 = mx1)
        cusp::multiply(A, pn, vn);

        //wn = A'*vn; (nxm * mx1 = nx1)
        cusp::multiply(At, vn, wn);
        
        //xn = softThreshold(pn + z - wn, tau);

        // at the end of these calls
        // threshold_vec = pn + z - wn 
        blas::copy( z, threshold_vec );
        blas::axpy( pn, threshold_vec, 1.0f);
        blas::axpy( wn, threshold_vec, -1.0f);

        // threshold threshold_vec and assign result to xn
        // for this we need to get pointers to the locations of the arrays
        // nthreads below should be number of gpu cores or half this number
        tvec_ptr = thrust::raw_pointer_cast(&threshold_vec[0]);
        tvec_ptr2 = thrust::raw_pointer_cast(&xn[0]);
        nthreads = 492;
        nblocks = (n + nthreads - 1)/nthreads;
        cudasoftThreshold<<<nblocks,nthreads>>>(n, tau, tvec_ptr, tvec_ptr2);

        //printf("in this iteration: blas::nrm2(xn) = %f\n", blas::nrm2(xn));
    // end iteration loop
    }
}  


/* perform CG solve for (A D)^T (A D) v = (A D)^T b with D being diagonal whose values are contained in T_global */
template <class LinearOperator, class Vector>
void cg_solve_diag(LinearOperator &A, LinearOperator &At, Vector &b, Vector &z, Vector &w0, Vector &wn, int * T_global, int num_iters_cg) {
    int i, nthreads, nblocks, iter_num, m, n, quit_loop = 0;
    float rtr, rtrold, dtp, alpha, beta;
    const float *cfloat_ptr;
    float *float_ptr1;
    float *float_ptr2;
    int *int_ptr1;
    int *int_ptr2;

    printf("start call..\n");

    // get dimensions
    m = A.num_rows;
    n = A.num_cols; 

    // set number of threads
    // this should be number of cores on the gpu or half this number
    // set number of blocks to use based on how n divides n threads
    nthreads = 192;
    nblocks = (n + nthreads - 1)/nthreads;

    cusp::array1d<float, cusp::device_memory> rn(n);
    cusp::array1d<float, cusp::device_memory> dn(n);
    cusp::array1d<float, cusp::device_memory> Ddn(n);
    cusp::array1d<float, cusp::device_memory> ADdn(m);
    cusp::array1d<float, cusp::device_memory> Atb(n);
    cusp::array1d<float, cusp::device_memory> AtADdn(n);
    cusp::array1d<float, cusp::device_memory> pn(n);

    Atb = z;

    // form rn = (A D)^T b = D*A^T*b
    //__global__ void cudaApplyD( int N, int *diag_elems, float* yn, float * result)
    float_ptr1 = thrust::raw_pointer_cast(&Atb[0]);
    float_ptr2 = thrust::raw_pointer_cast(&rn[0]);
    cudaApplyD<<<nblocks,nthreads>>>(n, T_global, float_ptr1, float_ptr2);

    
    // compute r^t*r dot product
    rtr = blas::dot(rn,rn);

    // start at initial guess
    wn = w0;

    for(iter_num=0; iter_num<num_iters_cg; iter_num++){
        printf("start iteration %d\n", iter_num);
        dn = rn;
       
        // form (A D)^T (A D) (A D)^T b = D A^T A D dn 

        // form D dn
        float_ptr1 = thrust::raw_pointer_cast(&dn[0]);
        float_ptr2 = thrust::raw_pointer_cast(&Ddn[0]);
        cudaApplyD<<<nblocks,nthreads>>>(n, T_global, float_ptr1, float_ptr2);

        printf("nrm2(Ddn) = %f\n", blas::nrm2(Ddn));
        // form A D dn
        cusp::multiply(A, Ddn, ADdn);

        

        // form At A D dn = A^T * ADdn
        cusp::multiply(At, ADdn, AtADdn);

        // pn = D* At A D dn
        float_ptr1 = thrust::raw_pointer_cast(&AtADdn[0]);
        float_ptr2 = thrust::raw_pointer_cast(&pn[0]);
        cudaApplyD<<<nblocks,nthreads>>>(n, T_global, float_ptr1, float_ptr2);

        // compute dot product between dn and pn
        dtp = blas::dot(dn,pn);

        alpha = rtr/dtp;

        //w = w + alpha * d;
        blas::axpy(dn, wn, alpha);

        //r = r - alpha * pn;
        blas::axpy(pn, rn, -alpha);

        rtrold = rtr;
        rtr = blas::dot(rn,rn);

        beta = rtr/rtrold;

        // d = r + beta * d;
        dn = rn;
        blas::axpy(rn, dn, beta);

        printf("at iteration %d norm(wn) = %f\n", iter_num, blas::nrm2(wn));
    } 
}


int main(void)
{
    // define some vars
    int i, m,n, num_iters = 10000;
    float val, maxz, tau, time_diff, percent_error;
    FILE *fp, *fp_out;
    char * line;
    time_t start_time, end_time;

    char *data_dir = (char *)malloc(1000*sizeof(char));
    char *A_file = (char *) malloc(1000*sizeof(char));
    char *b_file = (char *) malloc(1000*sizeof(char));
    char *x_file = (char *) malloc(1000*sizeof(char));

    strcpy(data_dir,"../data/system_data/well_conditioned_staircase/matrix_market/system1/");
    strcpy(A_file,data_dir);
    strcat(A_file,"/A.mtx");
    strcpy(b_file,data_dir);
    strcat(b_file,"/b.txt");
    strcpy(x_file,data_dir);
    strcat(x_file,"/x.txt");


    fp_out = fopen("log_run1.txt","w");
    printf("reading matrix from disk..\n");

    // read matrix from disk
    printf("reading matrix from disk at %s to host memory..\n", A_file);
    time(&start_time);
    cusp::coo_matrix<int, float, cusp::host_memory> Ah;
    cusp::io::read_matrix_market_file(Ah, A_file);
    time(&end_time);
    time_diff = difftime(end_time,start_time);
    printf("elapsed time: %f\n", time_diff);


    // we change to more efficient hyb sparse format on device
    printf("copying to device memory and changing format..\n");
    time(&start_time);
    cusp::hyb_matrix<int,float,cusp::device_memory> A = Ah;
    time(&end_time);
    time_diff = difftime(end_time,start_time);
    printf("elapsed time: %f\n", time_diff);


    // make the transpose
    printf("make transpose on host memory..\n");
    time(&start_time);
    cusp::coo_matrix<int, float, cusp::host_memory> Aht;
    cusp::transpose(Ah, Aht);
    time(&end_time);
    time_diff = difftime(end_time,start_time);
    printf("elapsed time: %f\n", time_diff);


    // we change to more efficient hyb sparse format on device
    printf("copying to device memory and changing format..\n");
    time(&start_time);
    cusp::hyb_matrix<int,float,cusp::device_memory> At = Aht;
    time(&end_time);
    time_diff = difftime(end_time,start_time);
    printf("elapsed time: %f\n", time_diff);

    
    // get matrix dimensions
    m = A.num_rows;
    n = A.num_cols;
    printf("m = %d and n = %d\n", m,n);

    // read vector x from disk to device
    printf("read x vector from %s\n", x_file);
    line = (char*)malloc(50*sizeof(char));
    fp = fopen(x_file,"r");
    fscanf(fp, "%s\n", line);
    n = atoi(line);
    cusp::array1d<float, cusp::device_memory> x(n);
    for(i=0; i<n; i++){
        fscanf(fp, "%s\n", line);
        val = atof(line);
        x[i] = val;
        printf("x[%d] = %f\n", i, val);
    }
    fclose(fp);

    // read vector b from disk to device
    printf("read b vector from %s\n", b_file);
    fp = fopen(b_file,"r");
    fscanf(fp, "%s\n", line);
    m = atoi(line);
    printf("m = %d\n",m);
    cusp::array1d<float, cusp::device_memory> b(m);
    for(i=0; i<m; i++){
        fscanf(fp, "%s\n", line);
        val = atof(line);
        b[i] = val;
        printf("b[%d] = %f\n", i, val);
    }
    fclose(fp);
    printf("freeing line..\n");
    free(line);
    

    printf("done reading..\n");
    printf("blas::nrm2(x) = %f\n", blas::nrm2(x));
    printf("blas::nrm2(b) = %f\n", blas::nrm2(b));


    printf("running fista algorithm..\n");

    // start timer
    time(&start_time);

    // set up more vars
    cusp::array1d<float, cusp::device_memory> x0(n);
    cusp::array1d<float, cusp::device_memory> xn(n);
    cusp::array1d<float, cusp::device_memory> dn(n);

    
    // calculate z = A^t*b (nxm * mx1 = nx1)
    cusp::array1d<float, cusp::device_memory> z(n);
    cusp::multiply(At, b, z);
    maxz = blas::nrmmax(z);

    // set array of taus and run the code...
    tau = maxz/1e2; 

    // init zero initial guess
    x0 = z;
    blas::scal(x0,0.0);

    // run algorithm 
    printf("running algorithm with tau = %f\n --->", tau);
    fistaAlgorithm(A, At, x0, b, z, num_iters, tau, xn );
    printf("algorithm finished. blas::nrm2(xn) = %f\n", blas::nrm2(xn));

    // calculate percent error
    //blas::copy( dn, xn );
    dn = xn;
    printf("after copy call: blas::nrm2(dn) = %f\n", blas::nrm2(dn));
    blas::axpy( x, dn, -1.0); // dn = xn - x
    printf("after axpy call: blas::nrm2(dn) = %f\n", blas::nrm2(dn));

    percent_error = 100*(blas::nrm2(dn))/(blas::nrm2(x));
    printf("percent error = %f\n", percent_error);

    // end timer
    time(&end_time); 
    time_diff = difftime(end_time,start_time);
    
    printf("Finished. elapsed time: %4.3f\n", time_diff);

    fprintf(fp_out,"done! elapsed time: %f and ||xn||_2 = %f\n", time_diff, blas::nrm2(xn));
    fprintf(fp_out,"percent error = %f\n", percent_error);
    fclose(fp_out);
    return 0;
}

