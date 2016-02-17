set(HEADER_FILES
    sopt_error.h sopt_l2.h   sopt_ran.h        sopt_tv.h
    sopt_image.h sopt_meas.h sopt_sara.h       sopt_types.h
    sopt_l1.h    sopt_prox.h sopt_sparsemat.h  sopt_utility.h
    sopt_wavelet.h "${PROJECT_BINARY_DIR}/include/sopt_config.h"
)

# Creates an overall configuration file
set(SOPT_BLAS_H "${BLAS_INCLUDE_FILENAME}")
configure_file(sopt_config.in.h "${PROJECT_BINARY_DIR}/include/sopt/sopt_config.h")

install(FILES ${HEADER_FILES} DESTINATION include/sopt COMPONENT dev)
