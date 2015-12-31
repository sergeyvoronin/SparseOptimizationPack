#
CFLAGS 	        = 
FFLAGS	        = 
CPPFLAGS        =
FPPFLAGS        =

PETSC_DIR=/opt/petsc_software/petsc-3.5.1
PETSC_ARCH=arch-linux2-c-debug

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

petsc_test1: petsc_test1.o chkopts
	-${CLINKER}  -o petsc_test1 petsc_test1.o  ${PETSC_MAT_LIB} 
	${RM} petsc_test1.o 


petsc_test_l1: petsc_test_l1.o petsc_l1/l1optmethods.o petsc_l1/wtransforms.o chkopts
	-${CLINKER}  -o petsc_test_l1 petsc_test_l1.o petsc_l1/l1optmethods.o petsc_l1/wtransforms.o ${PETSC_MAT_LIB} 
	${RM} petsc_test_l1.o petsc_l1/l1optmethods.o petsc_l1/wtransforms.o

