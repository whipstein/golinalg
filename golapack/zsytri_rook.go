package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrirook computes the inverse of a complex symmetric
// matrix A using the factorization A = U*D*U**T or A = L*D*L**T
// computed by ZSYTRF_ROOK.
func ZsytriRook(uplo mat.MatUplo, n int, a *mat.CMatrix, ipiv *[]int, work *mat.CVector) (info int, err error) {
	var upper bool
	var ak, akkp1, akp1, cone, czero, d, t, temp complex128
	var k, kp, kstep int

	cone = (1.0 + 0.0*1i)
	czero = (0.0 + 0.0*1i)

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
		gltest.Xerbla2("ZsytriRook", err)
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
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == czero {
				return
			}
		}
	} else {
		//        Lower triangular storage: examine D from top to bottom.
		for info = 1; info <= n; info++ {
			if (*ipiv)[info-1] > 0 && a.Get(info-1, info-1) == czero {
				return
			}
		}
	}
	info = 0

	if upper {
		//        Compute inv(A) from the factorization A = U*D*U**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = 1
	label30:
		;

		//        If K > N, exit from loop.
		if k > n {
			goto label40
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.Set(k-1, k-1, cone/a.Get(k-1, k-1))

			//           Compute column K of the inverse.
			if k > 1 {
				work.Copy(k-1, a.Off(0, k-1).CVector(), 1, 1)
				if err = Zsymv(uplo, k-1, -cone, a, work, 1, czero, a.Off(0, k-1).CVector(), 1); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(0, k-1).CVector().Dotu(k-1, work, 1, 1))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = a.Get(k-1, k)
			ak = a.Get(k-1, k-1) / t
			akp1 = a.Get(k, k) / t
			akkp1 = a.Get(k-1, k) / t
			d = t * (ak*akp1 - cone)
			a.Set(k-1, k-1, akp1/d)
			a.Set(k, k, ak/d)
			a.Set(k-1, k, -akkp1/d)

			//           Compute columns K and K+1 of the inverse.
			if k > 1 {
				work.Copy(k-1, a.Off(0, k-1).CVector(), 1, 1)
				if err = Zsymv(uplo, k-1, -cone, a, work, 1, czero, a.Off(0, k-1).CVector(), 1); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(0, k-1).CVector().Dotu(k-1, work, 1, 1))
				a.Set(k-1, k, a.Get(k-1, k)-a.Off(0, k).CVector().Dotu(k-1, a.Off(0, k-1).CVector(), 1, 1))
				work.Copy(k-1, a.Off(0, k).CVector(), 1, 1)
				if err = Zsymv(uplo, k-1, -cone, a, work, 1, czero, a.Off(0, k).CVector(), 1); err != nil {
					panic(err)
				}
				a.Set(k, k, a.Get(k, k)-a.Off(0, k).CVector().Dotu(k-1, work, 1, 1))
			}
			kstep = 2
		}

		if kstep == 1 {
			//           Interchange rows and columns K and IPIV(K) in the leading
			//           submatrix A(1:k+1,1:k+1)
			kp = (*ipiv)[k-1]
			if kp != k {
				if kp > 1 {
					a.Off(0, kp-1).CVector().Swap(kp-1, a.Off(0, k-1).CVector(), 1, 1)
				}
				a.Off(kp-1, kp).CVector().Swap(k-kp-1, a.Off(kp, k-1).CVector(), 1, a.Rows)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		} else {
			//           Interchange rows and columns K and K+1 with -IPIV(K) and
			//           -IPIV(K+1)in the leading submatrix A(1:k+1,1:k+1)
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp > 1 {
					a.Off(0, kp-1).CVector().Swap(kp-1, a.Off(0, k-1).CVector(), 1, 1)
				}
				a.Off(kp-1, kp).CVector().Swap(k-kp-1, a.Off(kp, k-1).CVector(), 1, a.Rows)

				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
				temp = a.Get(k-1, k)
				a.Set(k-1, k, a.Get(kp-1, k))
				a.Set(kp-1, k, temp)
			}

			k = k + 1
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp > 1 {
					a.Off(0, kp-1).CVector().Swap(kp-1, a.Off(0, k-1).CVector(), 1, 1)
				}
				a.Off(kp-1, kp).CVector().Swap(k-kp-1, a.Off(kp, k-1).CVector(), 1, a.Rows)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		}

		k = k + 1
		goto label30
	label40:
	} else {
		//        Compute inv(A) from the factorization A = L*D*L**T.
		//
		//        K is the main loop index, increasing from 1 to N in steps of
		//        1 or 2, depending on the size of the diagonal blocks.
		k = n
	label50:
		;

		//        If K < 1, exit from loop.
		if k < 1 {
			goto label60
		}

		if (*ipiv)[k-1] > 0 {
			//           1 x 1 diagonal block
			//
			//           Invert the diagonal block.
			a.Set(k-1, k-1, cone/a.Get(k-1, k-1))

			//           Compute column K of the inverse.
			if k < n {
				work.Copy(n-k, a.Off(k, k-1).CVector(), 1, 1)
				if err = Zsymv(uplo, n-k, -cone, a.Off(k, k), work, 1, czero, a.Off(k, k-1).CVector(), 1); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(k, k-1).CVector().Dotu(n-k, work, 1, 1))
			}
			kstep = 1
		} else {
			//           2 x 2 diagonal block
			//
			//           Invert the diagonal block.
			t = a.Get(k-1, k-1-1)
			ak = a.Get(k-1-1, k-1-1) / t
			akp1 = a.Get(k-1, k-1) / t
			akkp1 = a.Get(k-1, k-1-1) / t
			d = t * (ak*akp1 - cone)
			a.Set(k-1-1, k-1-1, akp1/d)
			a.Set(k-1, k-1, ak/d)
			a.Set(k-1, k-1-1, -akkp1/d)

			//           Compute columns K-1 and K of the inverse.
			if k < n {
				work.Copy(n-k, a.Off(k, k-1).CVector(), 1, 1)
				if err = Zsymv(uplo, n-k, -cone, a.Off(k, k), work, 1, czero, a.Off(k, k-1).CVector(), 1); err != nil {
					panic(err)
				}
				a.Set(k-1, k-1, a.Get(k-1, k-1)-a.Off(k, k-1).CVector().Dotu(n-k, work, 1, 1))
				a.Set(k-1, k-1-1, a.Get(k-1, k-1-1)-a.Off(k, k-1-1).CVector().Dotu(n-k, a.Off(k, k-1).CVector(), 1, 1))
				work.Copy(n-k, a.Off(k, k-1-1).CVector(), 1, 1)
				if err = Zsymv(uplo, n-k, -cone, a.Off(k, k), work, 1, czero, a.Off(k, k-1-1).CVector(), 1); err != nil {
					panic(err)
				}
				a.Set(k-1-1, k-1-1, a.Get(k-1-1, k-1-1)-a.Off(k, k-1-1).CVector().Dotu(n-k, work, 1, 1))
			}
			kstep = 2
		}

		if kstep == 1 {
			//           Interchange rows and columns K and IPIV(K) in the trailing
			//           submatrix A(k-1:n,k-1:n)
			kp = (*ipiv)[k-1]
			if kp != k {
				if kp < n {
					a.Off(kp, kp-1).CVector().Swap(n-kp, a.Off(kp, k-1).CVector(), 1, 1)
				}
				a.Off(kp-1, k).CVector().Swap(kp-k-1, a.Off(k, k-1).CVector(), 1, a.Rows)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		} else {
			//           Interchange rows and columns K and K-1 with -IPIV(K) and
			//           -IPIV(K-1) in the trailing submatrix A(k-1:n,k-1:n)
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp < n {
					a.Off(kp, kp-1).CVector().Swap(n-kp, a.Off(kp, k-1).CVector(), 1, 1)
				}
				a.Off(kp-1, k).CVector().Swap(kp-k-1, a.Off(k, k-1).CVector(), 1, a.Rows)

				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
				temp = a.Get(k-1, k-1-1)
				a.Set(k-1, k-1-1, a.Get(kp-1, k-1-1))
				a.Set(kp-1, k-1-1, temp)
			}

			k = k - 1
			kp = -(*ipiv)[k-1]
			if kp != k {
				if kp < n {
					a.Off(kp, kp-1).CVector().Swap(n-kp, a.Off(kp, k-1).CVector(), 1, 1)
				}
				a.Off(kp-1, k).CVector().Swap(kp-k-1, a.Off(k, k-1).CVector(), 1, a.Rows)
				temp = a.Get(k-1, k-1)
				a.Set(k-1, k-1, a.Get(kp-1, kp-1))
				a.Set(kp-1, kp-1, temp)
			}
		}

		k = k - 1
		goto label50
	label60:
	}

	return
}
