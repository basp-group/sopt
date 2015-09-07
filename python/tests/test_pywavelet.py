"""test cython binding of wavelets"""
def test_1D():
    import wavelets.wavelets as wv
    import numpy as np
    signal = np.ones((64,1), dtype="float64")
    coefficient = wv.dwt(signal, "DB4", 1)
    inv_signal = wv.dwt(coefficient, "DB4", 1, inverse = True)
    np.testing.assert_allclose(signal, inv_signal)



def test_2D():
    import wavelets.wavelets as wv
    import numpy as np
    db_name = ["DB1","DB2","DB3","DB4","DB5","DB6","DB7"]
    
    for ncol in [1, 4, 32, 64]:
        signal = np.random.random((32,ncol))
        for name in db_name:
            coefficient  = wv.dwt(signal,name,1)
            inv_signal = wv.dwt(coefficient, name, 1, inverse = True)
            np.testing.assert_allclose(signal, inv_signal)

def test_complex():
    import wavelets.wavelets as wv
    import numpy as np
    s_real = np.random.random((64,64))
    s_img = np.random.random((64,64))
    signal = s_real + s_img*1j
    coefficient = wv.dwt(signal, "DB4", 1)
    inv_signal = wv.dwt(coefficient, "DB4", 1, inverse = True)
    np.testing.assert_allclose(signal, inv_signal)

def test_1D_pywt():
    """compare the result of 1D DB1 direct transform 
    with pywt library"""
    import numpy as np
    import wavelets.wavelets as wv
    import pywt
    input = np.random.random(128)
    coefficient_sopt = wv.dwt(input,"DB1",1)
    cA_pywt, cD_pywt = pywt.dwt(input, "DB1")
    coefficient_pywt = np.concatenate((cA_pywt,-1*cD_pywt)).reshape(coefficient_sopt.shape)
    np.testing.assert_allclose(coefficient_sopt, coefficient_pywt)
    print coefficient_sopt-coefficient_pywt


