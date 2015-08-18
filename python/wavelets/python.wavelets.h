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
      typedef Eigen::Matrix
        <T,Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> t_pyMatrix;
      t_pyMatrix eigenMat(nrow, ncol);
      eigenMat =Eigen::Map<t_pyMatrix>(&pyMat[0],nrow,ncol);
      return(eigenMat);
    }


  template <class T>
    void direct(T *py_input, T* py_output, std::string name, t_uint level,
        const int nrow, const int ncol){
      auto const wavelets = sopt::wavelets::factory(name, level);
      auto eigen_input  = FromPyMat2EigenMat(py_input, nrow, ncol);
      auto coefficient = wavelets.direct(eigen_input);
      for(int i=0; i<nrow*ncol; ++i)
        py_output[i]=coefficient.data()[i];
    };

  template <class T>
    void direct(T *py_input, T* py_coefficient, T* py_output, std::string name,\
        t_uint level, const int nrow, const int ncol){
      auto const wavelets = sopt::wavelets::factory(name, level);
      auto eigen_output = FromPyMat2EigenMat(py_output, nrow, ncol);
      auto eigen_coefficient = FromPyMat2EigenMat(py_coefficient, nrow, ncol);
      wavelets.direct(eigen_output, eigen_coefficient);
      for(int i=0; i<nrow*ncol; ++i)
        py_output[i]=eigen_output.data()[i];
//      std::cout<<eigen_outputc<std::endl;
    };

  template <class T>
    void indirect(T *py_input, T *py_output, std::string name, t_uint level,\
        const int nrow, const int ncol){
      auto const wavelets = sopt::wavelets::factory(name, level);
      auto eigen_input  = FromPyMat2EigenMat(py_input, nrow, ncol);
      auto coefficient = wavelets.indirect(eigen_input);
      for(int i=0; i<nrow*ncol; ++i)
        py_output[i]=coefficient.data()[i];
    };
  
  template <class T>
    void indirect(T *py_input, T* py_coefficient, T* py_output, std::string name,\
        t_uint level, const int nrow, const int ncol){
      auto const wavelets = sopt::wavelets::factory(name, level);
      auto eigen_output = FromPyMat2EigenMat(py_output, nrow, ncol);
      auto eigen_coefficient = FromPyMat2EigenMat(py_coefficient, nrow, ncol);

      std::cout<<eigen_output<<std::endl;

      wavelets.indirect(eigen_output, eigen_coefficient);
      for(int i=0; i<nrow*ncol; ++i){
        py_output[i]=eigen_output.data()[i];
        //py_coefficient[i]=eigen_coefficient.data()[i];
      }
      std::cout<<eigen_output<<std::endl;

    };



}}
#endif
