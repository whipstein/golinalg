package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsptri computes the inverse of a complex symmetric indefinite matrix
// A in packed storage using the factorization A = U*D*U**T or
// A = L*D*L**T computed by ZSPTRF.
func Zsptri(uplo mat.MatUplo, n int, ap *mat.CVector, ipiv *[]int, work *mat.CVector) (info int, err error) {
	var upper bool
	var ak, akkp1, akp1, d, one, t, temp, zero complex128
	var j, k, kc, kcnext, kp, kpc, kstep, kx, npp int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	}
	if err != nil {
		gltest.Xerbla2("Zsptri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		kp = n * (n + 1) / 2
		for info = n; info >= 1; info-- {
			if (*ipiv)[info-1] > 0 && ap.Get(kp-1) == zero {
				return
			}
			kp = kp - info
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		kp = 1
		for info = 1; info <= n; info++ {
			if (*ipiv)[info-1] > 0 && ap.Get(kp-1) == zero {
				return
			}
			kp = kp + n - info + 1
		}
	}
	info = 0

	if upper {
		//        Compute inv(A) from the factorization A = U*D*U**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
		kc = 1
	label30:
		;

		//        If K > N, exit from loop.
		if k > n {
			goto label50
		}

		kcnext = kc + k
		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			ap.Set(kc+k-1-1, one/ap.Get(kc+k-1-1))

			//           Compute column K of the inverse.
			if k > 1 {
				work.Copy(k-1, ap.Off(kc-1), 1, 1)
				if err = Zspmv(uplo, k-1, -one, ap, work, 1, zero, ap.Off(kc-1), 1); err != nil {
					panic(err)
				}
				ap.Set(kc+k-1-1, ap.Get(kc+k-1-1)-ap.Off(kc-1).Dotu(k-1, work, 1, 1))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = ap.Get(kcnext + k - 1 - 1)
			ak = ap.Get(kc+k-1-1) / t
			akp1 = ap.Get(kcnext+k-1) / t
			akkp1 = ap.Get(kcnext+k-1-1) / t
			d = t * (ak*akp1 - one)
			ap.Set(kc+k-1-1, akp1/d)
			ap.Set(kcnext+k-1, ak/d)
			ap.Set(kcnext+k-1-1, -akkp1/d)

			//           Compute columns K and K+1 of the inverse.
			if k > 1 {
				work.Copy(k-1, ap.Off(kc-1), 1, 1)
				if err = Zspmv(uplo, k-1, -one, ap, work, 1, zero, ap.Off(kc-1), 1); err != nil {
					panic(err)
				}
				ap.Set(kc+k-1-1, ap.Get(kc+k-1-1)-ap.Off(kc-1).Dotu(k-1, work, 1, 1))
				ap.Set(kcnext+k-1-1, ap.Get(kcnext+k-1-1)-ap.Off(kcnext-1).Dotu(k-1, ap.Off(kc-1), 1, 1))
				work.Copy(k-1, ap.Off(kcnext-1), 1, 1)
				if err = Zspmv(uplo, k-1, -one, ap, work, 1, zero, ap.Off(kcnext-1), 1); err != nil {
					panic(err)
				}
				ap.Set(kcnext+k-1, ap.Get(kcnext+k-1)-ap.Off(kcnext-1).Dotu(k-1, work, 1, 1))
			}
			kstep = 2
			kcnext = kcnext + k + 1
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the leading
			//           submatrix A(1:k+1,1:k+1)
			kpc = (kp-1)*kp/2 + 1
			ap.Off(kpc-1).Swap(kp-1, ap.Off(kc-1), 1, 1)
			kx = kpc + kp - 1
			for j = kp + 1; j <= k-1; j++ {
				kx = kx + j - 1
				temp = ap.Get(kc + j - 1 - 1)
				ap.Set(kc+j-1-1, ap.Get(kx-1))
				ap.Set(kx-1, temp)
			}
			temp = ap.Get(kc + k - 1 - 1)
			ap.Set(kc+k-1-1, ap.Get(kpc+kp-1-1))
			ap.Set(kpc+kp-1-1, temp)
			if kstep == 2 {
				temp = ap.Get(kc + k + k - 1 - 1)
				ap.Set(kc+k+k-1-1, ap.Get(kc+k+kp-1-1))
				ap.Set(kc+k+kp-1-1, temp)
			}
		}

		k = k + kstep
		kc = kcnext
		goto label30
	label50:
	} else {
		//
		//        Compute inv(A) from the factorization A = L*D*L**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		npp = n * (n + 1) / 2
		k = n
		kc = npp
	label60:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label80
		}

		kcnext = kc - (n - k + 2)
		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			ap.Set(kc-1, one/ap.Get(kc-1))

			//           Compute column K of the inverse.
			if k < n {
				work.Copy(n-k, ap.Off(kc), 1, 1)
				if err = Zspmv(uplo, n-k, -one, ap.Off(kc+n-k), work, 1, zero, ap.Off(kc), 1); err != nil {
					panic(err)
				}
				ap.Set(kc-1, ap.Get(kc-1)-ap.Off(kc).Dotu(n-k, work, 1, 1))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = ap.Get(kcnext + 1 - 1)
			ak = ap.Get(kcnext-1) / t
			akp1 = ap.Get(kc-1) / t
			akkp1 = ap.Get(kcnext) / t
			d = t * (ak*akp1 - one)
			ap.Set(kcnext-1, akp1/d)
			ap.Set(kc-1, ak/d)
			ap.Set(kcnext, -akkp1/d)
			//
			//           Compute columns K-1 and K of the inverse.
			if k < n {
				work.Copy(n-k, ap.Off(kc), 1, 1)
				if err = Zspmv(uplo, n-k, -one, ap.Off(kc+(n-k+1)-1), work, 1, zero, ap.Off(kc), 1); err != nil {
					panic(err)
				}
				ap.Set(kc-1, ap.Get(kc-1)-ap.Off(kc).Dotu(n-k, work, 1, 1))
				ap.Set(kcnext, ap.Get(kcnext)-ap.Off(kcnext+2-1).Dotu(n-k, ap.Off(kc), 1, 1))
				work.Copy(n-k, ap.Off(kcnext+2-1), 1, 1)
				if err = Zspmv(uplo, n-k, -one, ap.Off(kc+(n-k+1)-1), work, 1, zero, ap.Off(kcnext+2-1), 1); err != nil {
					panic(err)
				}
				ap.Set(kcnext-1, ap.Get(kcnext-1)-ap.Off(kcnext+2-1).Dotu(n-k, work, 1, 1))
			}
			kstep = 2
			kcnext = kcnext - (n - k + 3)
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the trailing
			//           submatrix A(k-1:n,k-1:n)
			kpc = npp - (n-kp+1)*(n-kp+2)/2 + 1
			if kp < n {
				ap.Off(kpc).Swap(n-kp, ap.Off(kc+kp-k), 1, 1)
			}
			kx = kc + kp - k
			for j = k + 1; j <= kp-1; j++ {
				kx = kx + n - j + 1
				temp = ap.Get(kc + j - k - 1)
				ap.Set(kc+j-k-1, ap.Get(kx-1))
				ap.Set(kx-1, temp)
			}
			temp = ap.Get(kc - 1)
			ap.Set(kc-1, ap.Get(kpc-1))
			ap.Set(kpc-1, temp)
			if kstep == 2 {
				temp = ap.Get(kc - n + k - 1 - 1)
				ap.Set(kc-n+k-1-1, ap.Get(kc-n+kp-1-1))
				ap.Set(kc-n+kp-1-1, temp)
			}
		}

		k = k - kstep
		kc = kcnext
		goto label60
	label80:
	}

	return
}
