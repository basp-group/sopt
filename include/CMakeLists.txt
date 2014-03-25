set(HEADER_FILES
    sopt_error.h sopt_l2.h   sopt_ran.h        sopt_tv.h
    sopt_image.h sopt_meas.h sopt_sara.h       sopt_types.h
    sopt_l1.h    sopt_prox.h sopt_sparsemat.h  sopt_utility.h
    sopt_wavelet.h
)

install(FILES ${HEADER_FILES} DESTINATION include/sopt COMPONENT DEV)
