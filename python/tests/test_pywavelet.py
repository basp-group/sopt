


def test_1D():
    import wavelets.wavelets as wv
    import numpy as np
    signal = np.ones((64,1), dtype="float64")
    coefficient = wv.dwt(signal, "DB4", 1)
    inv_signal = wv.dwt(coefficient, "DB4", 1, inverse = True)
    np.testing.assert_allclose(signal, inv_signal)



def test_inverse_equal():
    import wavelets.wavelets as wv
    import numpy as np
    db_name = ["DB1","DB2","DB3","DB4","DB5","DB6","DB7"]
    
    for ncol in [1, 4, 32, 64]:
        signal = np.random.random((32,ncol))
        for name in db_name:
            coefficient  = wv.dwt(signal,name,1)
            inv_signal = wv.dwt(coefficient, name, 1, inverse = True)
            np.testing.assert_allclose(signal, inv_signal)

def test_complex_input():
    import wavelets.wavelets as wv
    import numpy as np
    s_real = np.random.random((64,64))
    s_img = np.random.random((64,64))
    signal = s_real + s_img*1j
    coefficient = wv.dwt(signal, "DB4", 1)
    inv_signal = wv.dwt(coefficient, "DB4", 1, inverse = True)
    np.testing.assert_allclose(signal, inv_signal)



test_inverse_equal()
test_1D()
test_complex_input()

