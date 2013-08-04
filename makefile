# ======== OPTIONS ========

#FFTW_INSTALLED = 0
FFTW_INSTALLED = 1


# ======== COMPILER ========

CC      = gcc
OPT	= -Wall -O3 -fopenmp\
          -DSOPT_VERSION=\"0.1\" \
          -DSOPT_BUILD=\"`git rev-parse HEAD`\"

ifeq ($(FFTW_INSTALLED),1)
  OPT += -DSOPT_FFTW_INSTALLED
endif


# ======== LINKS ========

PROGDIR    = ..

SOPTDIR  = $(PROGDIR)/sopt
SOPTLIB  = $(SOPTDIR)/lib
SOPTLIBNM= sopt
SOPTSRC  = $(SOPTDIR)/src/c
SOPTBIN  = $(SOPTDIR)/bin
SOPTOBJ  = $(SOPTSRC)
SOPTINC  = $(SOPTDIR)/include
SOPTDOC  = $(SOPTDIR)/doc

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
  FFTWDIR      = $(PROGDIR)/fftw-3.2.2_fPIC
endif
ifeq ($(UNAME), Darwin)
  FFTWDIR      = $(PROGDIR)/fftw
endif
FFTWINC	     = $(FFTWDIR)/include
FFTWLIB      = $(FFTWDIR)/lib
FFTWLIBNM    = fftw3

TIFFDIR      = $(PROGDIR)/tiff
TIFFINC	     = $(TIFFDIR)/include
TIFFLIB      = $(TIFFDIR)/lib
TIFFLIBNM    = tiff

ifeq ($(UNAME), Linux)
  MLAB		= /usr/local/MATLAB/R2011b
  MLABINC	= ${MLAB}/extern/include
  MLABLIB	= ${MLAB}/extern/lib

  MEXEXT	= mexa64
  MEX 		= ${MLAB}/bin/mex
  MEXFLAGS	= -cxx
endif
ifeq ($(UNAME), Darwin)
  MLAB		= /Applications/MATLAB_R2011b.app
  MLABINC	= ${MLAB}/extern/include
  MLABLIB	= ${MLAB}/extern/lib

  MEXEXT	= mexmaci64
  MEX 		= ${MLAB}/bin/mex
  MEXFLAGS	= -cxx
endif

SOPTSRCMAT	= $(SOPTDIR)/src/matlab
SOPTOBJMAT  	= $(SOPTSRCMAT)
SOPTOBJMEX  	= $(SOPTSRCMAT)


# ======== SOURCE LOCATIONS ========

vpath %.c $(SOPTSRC)
vpath %.h $(SOPTINC)
vpath %_mex.c $(SOPTSRCMAT)


# ======== FFFLAGS ========

FFLAGS  = -I$(SOPTINC) 
ifeq ($(FFTW_INSTALLED),1)
  FFLAGS += -I$(FFTWINC)
endif
FFLAGS += -I$(TIFFINC)


# ======== LDFLAGS ========

ifeq ($(UNAME), Linux)
  LDFLAGS = -static 
endif
ifeq ($(UNAME), Darwin)
  LDFLAGS = 
endif
LDFLAGS += -L$(SOPTLIB) -l$(SOPTLIBNM) 
ifeq ($(FFTW_INSTALLED),1)
  LDFLAGS += -L$(FFTWLIB) -l$(FFTWLIBNM)
endif
LDFLAGS += -L$(TIFFLIB) -l$(TIFFLIBNM)
LDFLAGS += -lm -lblas

LDFLAGSMEX = $(LDFLAGS) 


# ======== OBJECT FILES TO MAKE ========

SOPTOBJS = $(SOPTOBJ)/sopt_error.o         \
           $(SOPTOBJ)/sopt_l1.o            \
           $(SOPTOBJ)/sopt_tv.o            \
           $(SOPTOBJ)/sopt_prox.o          \
           $(SOPTOBJ)/sopt_utility.o       \
           $(SOPTOBJ)/sopt_image.o         \
           $(SOPTOBJ)/sopt_sparsemat.o     \
           $(SOPTOBJ)/sopt_ran.o           \
           $(SOPTOBJ)/sopt_meas.o          \
           $(SOPTOBJ)/sopt_l2.o            \
           $(SOPTOBJ)/sopt_sara.o          \
           $(SOPTOBJ)/sopt_wavelet.o    

SOPTHEADERS = sopt_error.h                 \
              sopt_types.h                 \
              sopt_l1.h                    \
              sopt_tv.h                    \
              sopt_prox.h                  \
              sopt_utility.h               \
              sopt_image.h                 \
              sopt_sparsemat.h             \
              sopt_ran.h                   \
              sopt_meas.h                  \
              sopt_l2.h                    \
              sopt_sara.h                  \
              sopt_wavelet.h

SOPTPROGS   = $(SOPTBIN)/sopt_about

SOPTOBJSMAT = $(SOPTOBJMAT)/sopt_solver_l1_mex.o \
              $(SOPTOBJMAT)/sopt_solver_tv_mex.o 

SOPTOBJSMEX = $(SOPTOBJMEX)/sopt_solver_l1_mex.$(MEXEXT) \
              $(SOPTOBJMEX)/sopt_solver_tv_mex.$(MEXEXT)        


# ======== MAKE RULES ========

$(SOPTOBJ)/%.o: %.c $(SOPTHEADERS)
	$(CC) $(OPT) $(FFLAGS) -c $< -o $@

.PHONY: default
default: lib

.PHONY: all
all: lib test prog demos

.PHONY: prog
prog: $(SOPTPROGS)
$(SOPTBIN)/%: %.c $(SOPTLIB)/lib$(SOPTLIBNM).a
	$(CC) $(OPT) $(FFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: test
test: $(SOPTBIN)/sopt_test
$(SOPTBIN)/sopt_test: $(SOPTOBJ)/sopt_test.c $(SOPTLIB)/lib$(SOPTLIBNM).a
	$(CC) $(OPT) $(FFLAGS) $< -o $@ $(LDFLAGS) 

.PHONY: demos
demos: $(SOPTBIN)/sopt_demo1 $(SOPTBIN)/sopt_demo2

.PHONY: demo2
demo2: $(SOPTBIN)/sopt_demo2
$(SOPTBIN)/sopt_demo2: $(SOPTOBJ)/sopt_demo2.c $(SOPTLIB)/lib$(SOPTLIBNM).a
	$(CC) $(OPT) $(FFLAGS) $< -o $@ $(LDFLAGS) 

.PHONY: demo1
demo1: $(SOPTBIN)/sopt_demo1
$(SOPTBIN)/sopt_demo1: $(SOPTOBJ)/sopt_demo1.c $(SOPTLIB)/lib$(SOPTLIBNM).a
	$(CC) $(OPT) $(FFLAGS) $< -o $@ $(LDFLAGS) 

.PHONY: runtest
runtest: test
	$(SOPTBIN)/sopt_test

.PHONY: cleantest
cleantest: 
	rm -r data/test/*


# Library

.PHONY: lib
lib: $(SOPTLIB)/lib$(SOPTLIBNM).a
$(SOPTLIB)/lib$(SOPTLIBNM).a: $(SOPTOBJS)
	ar -r $(SOPTLIB)/lib$(SOPTLIBNM).a $(SOPTOBJS)


# Matlab

$(SOPTOBJMAT)/%_mex.o: %_mex.c $(SOPTLIB)/lib$(SOPTLIBNM).a
	$(CC) $(OPT) $(FFLAGS) -c $< -o $@ -I$(MLABINC) 

$(SOPTOBJMEX)/%_mex.$(MEXEXT): $(SOPTOBJMAT)/%_mex.o $(SOPTLIB)/lib$(SOPTLIBNM).a
	$(MEX) $< -o $@ $(LDFLAGSMEX) $(MEXFLAGS) -L$(MLABLIB)

.PHONY: matlab
matlab: $(SOPTOBJSMEX)


# Documentation 

.PHONY: doc
doc:
	doxygen $(SOPTSRC)/doxygen.config
	$(MLAB)/bin/matlab -nodisplay -r "m2html('mfiles','src/matlab', 'htmldir','doc/matlab'); exit;"
.PHONY: cleandoc
cleandoc:
	rm -rf $(SOPTDOC)/c/*
	rm -rf $(SOPTDOC)/matlab/*


# Cleaning up

.PHONY: clean
clean:	tidy
	rm -f $(SOPTOBJ)/*.o
	rm -f $(SOPTLIB)/lib$(SOPTLIBNM).a
	rm -rf $(SOPTBIN)/*
	rm -f $(SOPTOBJMAT)/*.o
	rm -f $(SOPTOBJMEX)/*.$(MEXEXT)

.PHONY: tidy
tidy:
	rm -f *~
	rm -f $(SOPTSRC)/*~ 

.PHONY: cleanall
cleanall: clean cleandoc