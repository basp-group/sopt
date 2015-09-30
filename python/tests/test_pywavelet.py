"""test cython binding of wavelets"""


def test_1D():
    import sopt.wavelets as wv
    import numpy as np
    signal = np.random.rand(64) + 1e0
    coefficient = wv.dwt(signal, "DB4", 1)
    inv_signal = wv.dwt(coefficient, "DB4", 1, inverse=True)
    np.testing.assert_allclose(signal, inv_signal, rtol=1e-8)

def test_2D():
    import sopt.wavelets as wv
    import numpy as np
    db_name = ["DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"]
    for ncol in [1, 4, 32, 64]:
        signal = np.random.random((32, ncol))
        for name in db_name:
            coefficient = wv.dwt(signal, name, 1)
            inv_signal = wv.dwt(coefficient, name, 1, inverse=True)
            np.testing.assert_allclose(signal, inv_signal)


def test_complex():
    import sopt.wavelets as wv
    import numpy as np
    s_real = np.random.random((64, 64)) + 1e0
    s_img = np.random.random((64, 64)) + 1e0
    signal = s_real + s_img*1j
    coefficient = wv.dwt(signal, "DB4", 1)
    inv_signal = wv.dwt(coefficient, "DB4", 1, inverse=True)
    np.testing.assert_allclose(signal, inv_signal)


def test_1D_pywt():
    """compare the result of 1D DB1 direct transform
    with pywt library"""
    import numpy as np
    import sopt.wavelets as wv
    import pywt
    input = np.random.random(128) +  1e0
    coefficient_sopt = wv.dwt(input, "DB1", 1)
    cA_pywt, cD_pywt = pywt.dwt(input, "DB1")
    coefficient_pywt = np.concatenate(
                       (cA_pywt, -1*cD_pywt)).reshape(coefficient_sopt.shape)
    np.testing.assert_allclose(coefficient_sopt, coefficient_pywt)


def test_wrong_dims():
    from pytest import raises
    import sopt.wavelets as wv
    import numpy as np
    signal = np.random.random((32, 32, 32))
    with raises(ValueError):
        wv.dwt(signal, "DB1", 1)


def test_wrong_type():
    from pytest import raises
    import sopt.wavelets as wv
    import numpy as np
    signal = np.random.random((32, 32))
    with raises(ValueError):
        wv.dwt(signal.astype("S1"), "DB1", 1)


def test_noncontiguous():
    import sopt.wavelets as wv
    import numpy as np
    signal = np.random.random((32, )) + 1e0
    coeff_nc = wv.dwt(signal[::4], "DB1", 1)
    coeff_cont = wv.dwt(signal[::4].copy(), "DB1", 1)
    np.testing.assert_allclose(coeff_nc, coeff_cont)
