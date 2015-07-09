#
CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =
include ${PETSC_DIR}/conf/base

petsc_test_l1: petsc_test_l1.o petsc_l1/l1optmethods.o petsc_l1/wtransforms.o chkopts
	-${CLINKER}  -o petsc_test_l1 petsc_test_l1.o petsc_l1/l1optmethods.o petsc_l1/wtransforms.o ${PETSC_MAT_LIB} 
	${RM} petsc_test_l1.o petsc_l1/l1optmethods.o petsc_l1/wtransforms.o

