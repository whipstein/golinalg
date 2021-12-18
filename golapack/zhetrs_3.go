package golapack

import (
	"fmt"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrs3 solves a system of linear equations A * X = B with a complex
// Hermitian matrix A using the factorization computed
// by ZHETRF_RK or ZHETRF_BK:
//
//    A = P*U*D*(U**H)*(P**T) or A = P*L*D*(L**H)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**H (or L**H) is the conjugate of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is Hermitian and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This algorithm is using Level 3 BLAS.
func Zhetrs3(uplo mat.MatUplo, n, nrhs int, a *mat.CMatrix, e *mat.CVector, ipiv *[]int, b *mat.CMatrix) (err error) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one complex128
	var s float64
	var i, j, k, kp int

	one = (1.0 + 0.0*1i)

	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if nrhs < 0 {
		err = fmt.Errorf("nrhs < 0: nrhs=%v", nrhs)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zhetrs3", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Begin Upper
		//
		//        Solve A*X = B, where A = U*D*U**H.
		//
		//        P**T * B
		//
		//        Interchange rows K and IPIV(K) of matrix B in the same order
		//        that the formation order of IPIV(I) vector for Upper case.
		//
		//        (We can do the simple loop over IPIV with decrement -1,
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = n; k >= 1; k-- {
			kp = abs((*ipiv)[k-1])
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
		}

		//        Compute (U \P**T * B) -> B    [ (U \P**T * B) ]
		if err = b.Trsm(Left, Upper, NoTrans, Unit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Compute D \ B -> B   [ D \ (U \P**T * B) ]
		i = n
		for i >= 1 {
			if (*ipiv)[i-1] > 0 {
				s = real(one) / a.GetRe(i-1, i-1)
				b.Off(i-1, 0).CVector().Dscal(nrhs, s, b.Rows)
			} else if i > 1 {
				akm1k = e.Get(i - 1)
				akm1 = a.Get(i-1-1, i-1-1) / akm1k
				ak = a.Get(i-1, i-1) / cmplx.Conj(akm1k)
				denom = akm1*ak - one
				for j = 1; j <= nrhs; j++ {
					bkm1 = b.Get(i-1-1, j-1) / akm1k
					bk = b.Get(i-1, j-1) / cmplx.Conj(akm1k)
					b.Set(i-1-1, j-1, (ak*bkm1-bk)/denom)
					b.Set(i-1, j-1, (akm1*bk-bkm1)/denom)
				}
				i = i - 1
			}
			i = i - 1
		}

		//        Compute (U**H \ B) -> B   [ U**H \ (D \ (U \P**T * B) ) ]
		if err = b.Trsm(Left, Upper, ConjTrans, Unit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        P * B  [ P * (U**H \ (D \ (U \P**T * B) )) ]
		//
		//        Interchange rows K and IPIV(K) of matrix B in reverse order
		//        from the formation order of IPIV(I) vector for Upper case.
		//
		//        (We can do the simple loop over IPIV with increment 1,
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = 1; k <= n; k++ {
			kp = abs((*ipiv)[k-1])
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
		}

	} else {
		//        Begin Lower
		//
		//        Solve A*X = B, where A = L*D*L**H.
		//
		//        P**T * B
		//        Interchange rows K and IPIV(K) of matrix B in the same order
		//        that the formation order of IPIV(I) vector for Lower case.
		//
		//        (We can do the simple loop over IPIV with increment 1,
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = 1; k <= n; k++ {
			kp = abs((*ipiv)[k-1])
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
		}

		//        Compute (L \P**T * B) -> B    [ (L \P**T * B) ]
		if err = b.Trsm(Left, Lower, NoTrans, Unit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        Compute D \ B -> B   [ D \ (L \P**T * B) ]
		i = 1
		for i <= n {
			if (*ipiv)[i-1] > 0 {
				s = real(one) / a.GetRe(i-1, i-1)
				b.Off(i-1, 0).CVector().Dscal(nrhs, s, b.Rows)
			} else if i < n {
				akm1k = e.Get(i - 1)
				akm1 = a.Get(i-1, i-1) / cmplx.Conj(akm1k)
				ak = a.Get(i, i) / akm1k
				denom = akm1*ak - one
				for j = 1; j <= nrhs; j++ {
					bkm1 = b.Get(i-1, j-1) / cmplx.Conj(akm1k)
					bk = b.Get(i, j-1) / akm1k
					b.Set(i-1, j-1, (ak*bkm1-bk)/denom)
					b.Set(i, j-1, (akm1*bk-bkm1)/denom)
				}
				i = i + 1
			}
			i = i + 1
		}

		//        Compute (L**H \ B) -> B   [ L**H \ (D \ (L \P**T * B) ) ]
		if err = b.Trsm(Left, Lower, ConjTrans, Unit, n, nrhs, one, a); err != nil {
			panic(err)
		}

		//        P * B  [ P * (L**H \ (D \ (L \P**T * B) )) ]
		//
		//        Interchange rows K and IPIV(K) of matrix B in reverse order
		//        from the formation order of IPIV(I) vector for Lower case.
		//
		//        (We can do the simple loop over IPIV with decrement -1,
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = n; k >= 1; k-- {
			kp = abs((*ipiv)[k-1])
			if kp != k {
				b.Off(kp-1, 0).CVector().Swap(nrhs, b.Off(k-1, 0).CVector(), b.Rows, b.Rows)
			}
		}

		//        END Lower
	}

	return
}
