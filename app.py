try:
    import numpy as np
    print("Numpy berhasil diimpor!")

    # Cek operasi dasar
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("Penjumlahan array:", a + b)

except ImportError:
    print("Numpy belum terinstall. Silakan install dengan 'pip install numpy'")
