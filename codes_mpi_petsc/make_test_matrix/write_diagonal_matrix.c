/*
 writes diagonal(2,...,2) matrix in Petsc format to disk; assumes 64 bit ints

petsc format:
   int    MAT_FILE_CLASSID

   int    number of rows

   int    number of columns

   int    total number of nonzeros

   int    *number nonzeros in each row

   int    *column indices of all nonzeros (starting index is zero)

   PetscScalar *values of all nonzeros
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_WIDTH          500
#define LITTLE_ENDIAN_SYS   0
#define BIG_ENDIAN_SYS      1
#define MAT_FILE_CLASSID 1211216 

#define DoByteSwap(x) ByteSwap((unsigned char *) &x, sizeof(x))

typedef long long int PetscInt;
typedef double PetscScalar;

// function prototypes
int machineEndianness();
void ByteSwap(unsigned char * b, int n);
void writeDiagonalMatrixInPetscFormat(char *petsc_full_filename);

int main(int argc,char **args){
    char *petsc_full_filename = "Mat1.petsc";
    printf("starting processing %s --->\n", petsc_full_filename);
    writeDiagonalMatrixInPetscFormat(petsc_full_filename);
    printf("finished writing to: %s\n", petsc_full_filename);
    return 0;
}


void writeDiagonalMatrixInPetscFormat(char *petsc_full_filename){

    // vars for PETSc
    FILE *io_out_petsc;
    int ind, num_vals;
    PetscInt i, j, r, row_num, col_num, N, matNumRows, matNumCols, matNumNNZ, matFileCookie;
    PetscScalar entry_val = 2.0;
    PetscInt * num_nnzs;
    PetscInt * col_nums;
    PetscInt * num_nnzs_per_row;
    size_t one = 1;
    int mySystemType = machineEndianness();

    PetscInt *temp_PetscInt_bswp_val;
    PetscScalar *temp_PetscScalar_bswp_val;

    printf("checking machine endianness\n");
    if( mySystemType == BIG_ENDIAN_SYS ){
        printf("machine is big endian\n");
    }else{
        printf("machine is little endian\n");
    }

    // set dims
    N = 1000;
    matNumRows = N;
    matNumCols = N;
    matNumNNZ = N;


    // write PETSc file ------>
    // set up filename
    printf("writing to %s\n", petsc_full_filename);
    io_out_petsc = fopen(petsc_full_filename,"w");

    // write cookie and rest of header
    printf("writing header..\n");
    matFileCookie = MAT_FILE_CLASSID;
    if( mySystemType == BIG_ENDIAN_SYS ){
        fwrite (&matFileCookie,sizeof(PetscInt),one,io_out_petsc);
        fwrite (&matNumRows,sizeof(PetscInt),one,io_out_petsc);
        fwrite (&matNumCols,sizeof(PetscInt),one,io_out_petsc);
        fwrite (&matNumNNZ,sizeof(PetscInt),one,io_out_petsc);
    }
    else{ // swap stuff to BIG_ENDIAN format
        DoByteSwap(matFileCookie); 
        fwrite(&matFileCookie,sizeof(PetscInt),one,io_out_petsc);
        DoByteSwap(matNumRows);
        fwrite(&matNumRows,sizeof(PetscInt),one,io_out_petsc);
        DoByteSwap(matNumCols);
        fwrite(&matNumCols,sizeof(PetscInt),one,io_out_petsc); 
        DoByteSwap(matNumNNZ);
        fwrite(&matNumNNZ,sizeof(PetscInt),one,io_out_petsc);
    }

    // number of nonzeros in each row
    printf("writing nnzs per row..\n");
    for(j=0; j<N; j++){
        temp_PetscInt_bswp_val = (PetscInt *)malloc(sizeof(PetscInt));
        *temp_PetscInt_bswp_val = (PetscInt) 1;
        DoByteSwap(temp_PetscInt_bswp_val[0]);
        fwrite (temp_PetscInt_bswp_val,sizeof(PetscInt),one,io_out_petsc);
        free(temp_PetscInt_bswp_val);
    }        

    // column indices on nonzeros 
    printf("writing column indices of nonzeros..\n");
    for(j=0; j<N; j++){
        // write j
        temp_PetscInt_bswp_val = (PetscInt *)malloc(sizeof(PetscInt));
        *temp_PetscInt_bswp_val = (PetscInt) j;
        DoByteSwap(temp_PetscInt_bswp_val[0]);
        fwrite (temp_PetscInt_bswp_val,sizeof(PetscInt),one,io_out_petsc);
        free(temp_PetscInt_bswp_val);
    }

    // values of the nonzeros 
    printf("writing values of nonzeros..\n");
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            temp_PetscScalar_bswp_val = (PetscScalar *)malloc(sizeof(PetscScalar));
            *temp_PetscScalar_bswp_val = entry_val; // constant value on diagonal
            DoByteSwap(temp_PetscScalar_bswp_val[0]);
            fwrite(temp_PetscScalar_bswp_val,sizeof(PetscScalar),one,io_out_petsc);
            free(temp_PetscScalar_bswp_val);
        }
    }

    printf("closing file..\n");
    fclose(io_out_petsc);
}


int machineEndianness(){
   long int i = 1;
   const char *p = (const char *) &i;
        /* check if lowest address contains the least significant byte */
   if (p[0] == 1)
      return LITTLE_ENDIAN_SYS;
   else
      return BIG_ENDIAN_SYS;
}


void ByteSwap(unsigned char * b, int n){
   register int i = 0;
   register int j = n-1;
   unsigned char temp;
   while (i<j){
      /* swap(b[i], b[j]) */
          temp = b[i];
          b[i] = b[j];
          b[j] = temp;
      i++, j--;
   }
}

