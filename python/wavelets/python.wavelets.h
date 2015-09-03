#ifndef SOPT_TESTME
#define  SOPT_TESTME
#include <Eigen/Core>
#include <iostream>
#include <wavelets.h>
#include <types.h>

namespace sopt { namespace pyWavelets{
  template<class T>
    Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FromPyMat2EigenMat(T *pyMat, int nrow, int ncol){
      typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> t_pyMatrix;
      t_pyMatrix eigenMat(nrow, ncol);
      eigenMat = Eigen::Map<t_pyMatrix>(&pyMat[0], nrow, ncol);
      return(eigenMat);
    }

  //! direct transform of 2D data
  template <class T>
    void direct(T *py_input, T* py_output, std::string name, t_uint level,
        const int nrow, const int ncol){
      auto const wavelets = sopt::wavelets::factory(name, level);
      auto eigen_input  = FromPyMat2EigenMat(py_input, nrow, ncol);
      if(eigen_input.cols() == 1){
        auto coefficient = wavelets.direct(eigen_input.col(0));
        for(int i=0; i<nrow*ncol; ++i)
          py_output[i]=coefficient.data()[i];
      }
      else{
        auto coefficient = wavelets.direct(eigen_input);
        for(int i=0; i<nrow*ncol; ++i)
          py_output[i]=coefficient.data()[i];
      }
    };
  
  template <class T>
    void indirect(T *py_input, T *py_output, std::string name, t_uint level,\
        const int nrow, const int ncol){
      auto const wavelets = sopt::wavelets::factory(name, level);
      auto eigen_input  = FromPyMat2EigenMat(py_input, nrow, ncol);
      if(eigen_input.cols() == 1){
        auto coefficient = wavelets.indirect(eigen_input.col(0));
        for(int i=0; i<nrow*ncol; ++i)
          py_output[i]=coefficient.data()[i];
      }
      else{
        auto coefficient = wavelets.indirect(eigen_input);
        for(int i=0; i<nrow*ncol; ++i)
          py_output[i]=coefficient.data()[i];
      }
    };
}}


#endif
