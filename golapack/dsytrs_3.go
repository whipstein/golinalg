package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dsytrs3 solves a system of linear equations A * X = B with a real
// symmetric matrix A using the factorization computed
// by DSYTRF_RK or DSYTRF_BK:
//
//    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This algorithm is using Level 3 BLAS.
func Dsytrs3(uplo mat.MatUplo, n, nrhs int, a *mat.Matrix, e *mat.Vector, ipiv *[]int, b *mat.Matrix) (info int, err error) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one float64
	var i, j, k, kp int

	one = 1.0

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
		gltest.Xerbla2("Dsytrs3", err)
		return
	}

	//     Quick return if possible
	if n == 0 || nrhs == 0 {
		return
	}

	if upper {
		//        Begin Upper
		//
		//        Solve A*X = B, where A = U*D*U**T.
		//
		//        P**T * B
		//
		//        Interchange rows K and IPIV(K) of matrix B in the same order
		//        that the formation order of IPIV(I) vector for Upper case.
		//
		//        (We can do the simple loop over IPIV with decrement -1,
		//        since the ABS value of IPIV( I ) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = n; k >= 1; k-- {
			kp = abs((*ipiv)[k-1])
			if kp != k {
				goblas.Dswap(nrhs, b.Vector(k-1, 0), b.Vector(kp-1, 0))
			}
		}

		//        Compute (U \P**T * B) -> B    [ (U \P**T * B) ]
		if err = goblas.Dtrsm(Left, Upper, NoTrans, Unit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        Compute D \ B -> B   [ D \ (U \P**T * B) ]
		i = n
		for i >= 1 {
			if (*ipiv)[i-1] > 0 {
				goblas.Dscal(nrhs, one/a.Get(i-1, i-1), b.Vector(i-1, 0))
			} else if i > 1 {
				akm1k = e.Get(i - 1)
				akm1 = a.Get(i-1-1, i-1-1) / akm1k
				ak = a.Get(i-1, i-1) / akm1k
				denom = akm1*ak - one
				for j = 1; j <= nrhs; j++ {
					bkm1 = b.Get(i-1-1, j-1) / akm1k
					bk = b.Get(i-1, j-1) / akm1k
					b.Set(i-1-1, j-1, (ak*bkm1-bk)/denom)
					b.Set(i-1, j-1, (akm1*bk-bkm1)/denom)
				}
				i = i - 1
			}
			i = i - 1
		}

		//        Compute (U**T \ B) -> B   [ U**T \ (D \ (U \P**T * B) ) ]
		if err = goblas.Dtrsm(Left, Upper, Trans, Unit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        P * B  [ P * (U**T \ (D \ (U \P**T * B) )) ]
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
				goblas.Dswap(nrhs, b.Vector(k-1, 0), b.Vector(kp-1, 0))
			}
		}

	} else {
		//        Begin Lower
		//
		//        Solve A*X = B, where A = L*D*L**T.
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
				goblas.Dswap(nrhs, b.Vector(k-1, 0), b.Vector(kp-1, 0))
			}
		}

		//        Compute (L \P**T * B) -> B    [ (L \P**T * B) ]
		if err = goblas.Dtrsm(Left, Lower, NoTrans, Unit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        Compute D \ B -> B   [ D \ (L \P**T * B) ]
		i = 1
		for i <= n {
			if (*ipiv)[i-1] > 0 {
				goblas.Dscal(nrhs, one/a.Get(i-1, i-1), b.Vector(i-1, 0))
			} else if i < n {
				akm1k = e.Get(i - 1)
				akm1 = a.Get(i-1, i-1) / akm1k
				ak = a.Get(i, i) / akm1k
				denom = akm1*ak - one
				for j = 1; j <= nrhs; j++ {
					bkm1 = b.Get(i-1, j-1) / akm1k
					bk = b.Get(i, j-1) / akm1k
					b.Set(i-1, j-1, (ak*bkm1-bk)/denom)
					b.Set(i, j-1, (akm1*bk-bkm1)/denom)
				}
				i = i + 1
			}
			i = i + 1
		}

		//        Compute (L**T \ B) -> B   [ L**T \ (D \ (L \P**T * B) ) ]
		if err = goblas.Dtrsm(Left, Lower, Trans, Unit, n, nrhs, one, a, b); err != nil {
			panic(err)
		}

		//        P * B  [ P * (L**T \ (D \ (L \P**T * B) )) ]
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
				goblas.Dswap(nrhs, b.Vector(k-1, 0), b.Vector(kp-1, 0))
			}
		}

		//        END Lower
	}

	return
}
