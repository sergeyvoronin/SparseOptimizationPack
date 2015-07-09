#include "petscksp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* extra includes for rand() and getcwd() */
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifndef L1OPTMETHODS_H
#define L1OPTMETHODS_H

typedef struct {
    PetscInt numiters;
    PetscScalar * residuals;
    PetscScalar * Js;
    PetscInt * num_nnzs;
    PetscScalar * percent_errors;
    PetscScalar * wn_norms;
    PetscScalar * wn_norms_wavelet;
    PetscScalar * xn_percent_diffs;
    PetscScalar * wn_percent_diffs;
} analysisStruct;

void setWaveletParams(PetscInt *params);
void setMonitorParams(PetscInt dump_frequency, char *output_dir);
void setIterStartNumber(PetscInt last_iter);
void setMonitorState(PetscInt flag);
PetscErrorCode monitorSolution(Vec xn, PetscInt iter_num, PetscScalar tau, char *method);
void setAnalysisOptions(PetscInt maxiters);

PetscErrorCode thresholdedLandweber(Mat A, Vec b, Vec true_solution, PetscScalar tau, Vec x0, PetscScalar TOL, PetscInt maxiters, Vec * output, PetscInt * numiters, analysisStruct * analysis_out);

PetscErrorCode thresholdedFista(Mat A, Vec b, PetscScalar tau, Vec x0, PetscScalar TOL, PetscInt maxiters, Vec * output, PetscInt * numiters, PetscScalar * residuals, PetscScalar * Js);

PetscErrorCode thresholded_coordinate_descent(Mat A, Vec b, Vec true_solution, PetscScalar tau, Vec x0, PetscScalar TOL, PetscInt maxiters, Vec * output, PetscInt * numiters, analysisStruct * analysis_out);

PetscErrorCode computeNumNNZs(Vec input, PetscInt *num_nnz, PetscScalar TOL);
PetscErrorCode softThreshold(Vec input, Vec *output, PetscScalar tau);
PetscErrorCode hardThreshold(Vec input, Vec *output, PetscScalar tau);
PetscErrorCode softThreshold2(Vec input, Vec *output, PetscScalar theta, PetscScalar tau);

PetscErrorCode getLargestSingularValue(Mat A, PetscInt niters, PetscScalar *sval);

PetscErrorCode projectOnL1Ball(Vec input, Vec *output, PetscScalar R);
int compfcn(const void *p, const void *q);


#endif

