package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetri computes the inverse of a complex Hermitian indefinite matrix
// A using the factorization A = U*D*U**H or A = L*D*L**H computed by
// ZHETRF.
func Zhetri(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int, work *mat.CVector) (info int, err error) {
	var upper bool
	var akkp1, cone, temp, zero complex128
	var ak, akp1, d, one, t float64
	var j, k, kp, kstep int

	one = 1.0
	cone = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zhetri", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Check that the diagonal matrix D is nonsingular.
	if upper {
		//        Upper triangular storage: examine D from bottom to top
		for info = n; info >= 1; info-- {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == zero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for info = 1; info <= n; info++ {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == zero {
				return
			}
		}
	}
	info = 0

	if upper {
		//        Compute inv(A) from the factorization A = U*D*U**H.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
	label30:
		;

		//        If K > N, exit from loop.
		if k > n {
			goto label50
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.SetRe(k-1, k-1, one/a.GetRe(k-1, k-1))

			//           Compute column K of the inverse.
			if k > 1 {
				goblas.Zcopy(k-1, a.CVector(0, k-1, 1), work.Off(0, 1))
				if err = goblas.Zhemv(uplo, k-1, -cone, a, work.Off(0, 1), zero, a.CVector(0, k-1, 1)); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-complex(real(goblas.Zdotc(k-1, work.Off(0, 1), a.CVector(0, k-1, 1))), 0))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = a.GetMag(k-1, k)
			ak = a.GetRe(k-1, k-1) / t
			akp1 = a.GetRe(k, k) / t
			akkp1 = a.Get(k-1, k) / complex(t, 0)
			d = t * (ak*akp1 - one)
			a.SetRe(k-1, k-1, akp1/d)
			a.SetRe(k, k, ak/d)
			a.Set(k-1, k, -akkp1/complex(d, 0))

			//           Compute columns K and K+1 of the inverse.
			if k > 1 {
				goblas.Zcopy(k-1, a.CVector(0, k-1, 1), work.Off(0, 1))
				if err = goblas.Zhemv(uplo, k-1, -cone, a, work.Off(0, 1), zero, a.CVector(0, k-1, 1)); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-complex(real(goblas.Zdotc(k-1, work.Off(0, 1), a.CVector(0, k-1, 1))), 0))
				a.Set(k-1, k, a.Get(k-1, k)-goblas.Zdotc(k-1, a.CVector(0, k-1, 1), a.CVector(0, k, 1)))
				goblas.Zcopy(k-1, a.CVector(0, k, 1), work.Off(0, 1))
				if err = goblas.Zhemv(uplo, k-1, -cone, a, work.Off(0, 1), zero, a.CVector(0, k, 1)); err != nil {
					panic(err)
				}
				a.Set(k, k, a.Get(k, k)-complex(real(goblas.Zdotc(k-1, work.Off(0, 1), a.CVector(0, k, 1))), 0))
			}
			kstep = 2
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the leading
			//           submatrix A(1:k+1,1:k+1)
			goblas.Zswap(kp-1, a.CVector(0, k-1, 1), a.CVector(0, kp-1, 1))
			for j = kp + 1; j <= k-1; j++ {
				temp = a.GetConj(j-1, k-1)
				a.Set(j-1, k-1, a.GetConj(kp-1, j-1))
				a.Set(kp-1, j-1, temp)
			}
			a.Set(kp-1, k-1, a.GetConj(kp-1, k-1))
			temp = a.Get(k-1, k-1)
			a.Set(k-1, k-1, a.Get(kp-1, kp-1))
			a.Set(kp-1, kp-1, temp)
			if kstep == 2 {
				temp = a.Get(k-1, k)
				a.Set(k-1, k, a.Get(kp-1, k))
				a.Set(kp-1, k, temp)
			}
		}

		k = k + kstep
		goto label30
	label50:
	} else {
		//        Compute inv(A) from the factorization A = L*D*L**H.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = n
	label60:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label80
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.Set(k-1, k-1, complex(one/a.GetRe(k-1, k-1), 0))

			//           Compute column K of the inverse.
			if k < n {
				goblas.Zcopy(n-k, a.CVector(k, k-1, 1), work.Off(0, 1))
				if err = goblas.Zhemv(uplo, n-k, -cone, a.Off(k, k), work.Off(0, 1), zero, a.CVector(k, k-1, 1)); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-complex(real(goblas.Zdotc(n-k, work.Off(0, 1), a.CVector(k, k-1, 1))), 0))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = a.GetMag(k-1, k-1-1)
			ak = a.GetRe(k-1-1, k-1-1) / t
			akp1 = a.GetRe(k-1, k-1) / t
			akkp1 = a.Get(k-1, k-1-1) / complex(t, 0)
			d = t * (ak*akp1 - one)
			a.SetRe(k-1-1, k-1-1, akp1/d)
			a.SetRe(k-1, k-1, ak/d)
			a.Set(k-1, k-1-1, -akkp1/complex(d, 0))

			//           Compute columns K-1 and K of the inverse.
			if k < n {
				goblas.Zcopy(n-k, a.CVector(k, k-1, 1), work.Off(0, 1))
				if err = goblas.Zhemv(uplo, n-k, -cone, a.Off(k, k), work.Off(0, 1), zero, a.CVector(k, k-1, 1)); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-complex(real(goblas.Zdotc(n-k, work.Off(0, 1), a.CVector(k, k-1, 1))), 0))
				a.Set(k-1, k-1-1, a.Get(k-1, k-1-1)-goblas.Zdotc(n-k, a.CVector(k, k-1, 1), a.CVector(k, k-1-1, 1)))
				goblas.Zcopy(n-k, a.CVector(k, k-1-1, 1), work.Off(0, 1))
				if err = goblas.Zhemv(uplo, n-k, -cone, a.Off(k, k), work.Off(0, 1), zero, a.CVector(k, k-1-1, 1)); err != nil {
					panic(err)
				}
				a.Set(k-1-1, k-1-1, a.Get(k-1-1, k-1-1)-complex(real(goblas.Zdotc(n-k, work.Off(0, 1), a.CVector(k, k-1-1, 1))), 0))
			}
			kstep = 2
		}

		kp = abs((*ipiv)[k-1])
		if kp != k {
			//           Interchange rows and columns K and KP in the trailing
			//           submatrix A(k-1:n,k-1:n)
			if kp < n {
				goblas.Zswap(n-kp, a.CVector(kp, k-1, 1), a.CVector(kp, kp-1, 1))
			}
			for j = k + 1; j <= kp-1; j++ {
				temp = a.GetConj(j-1, k-1)
				a.Set(j-1, k-1, a.GetConj(kp-1, j-1))
				a.Set(kp-1, j-1, temp)
			}
			a.Set(kp-1, k-1, a.GetConj(kp-1, k-1))
			temp = a.Get(k-1, k-1)
			a.Set(k-1, k-1, a.Get(kp-1, kp-1))
			a.Set(kp-1, kp-1, temp)
			if kstep == 2 {
				temp = a.Get(k-1, k-1-1)
				a.Set(k-1, k-1-1, a.Get(kp-1, k-1-1))
				a.Set(kp-1, k-1-1, temp)
			}
		}

		k = k - kstep
		goto label60
	label80:
	}

	return
}
