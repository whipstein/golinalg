package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrs3 solves a system of linear equations A * X = B with a complex
// symmetric matrix A using the factorization computed
// by ZSYTRF_RK or ZSYTRF_BK:
//
//    A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
//
// where U (or L) is unit upper (or lower) triangular matrix,
// U**T (or L**T) is the transpose of U (or L), P is a permutation
// matrix, P**T is the transpose of P, and D is symmetric and block
// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
//
// This algorithm is using Level 3 BLAS.
func Zsytrs3(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, e *mat.CVector, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var ak, akm1, akm1k, bk, bkm1, denom, one complex128
	var i, j, k, kp int
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -9
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYTRS_3"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
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
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = (*n); k >= 1; k-- {
			kp = absint((*ipiv)[k-1])
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}
		}

		//        Compute (U \P**T * B) -> B    [ (U \P**T * B) ]
		err = goblas.Ztrsm(Left, Upper, NoTrans, Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Compute D \ B -> B   [ D \ (U \P**T * B) ]
		i = (*n)
		for i >= 1 {
			if (*ipiv)[i-1] > 0 {
				goblas.Zscal(*nrhs, one/a.Get(i-1, i-1), b.CVector(i-1, 0), *ldb)
			} else if i > 1 {
				akm1k = e.Get(i - 1)
				akm1 = a.Get(i-1-1, i-1-1) / akm1k
				ak = a.Get(i-1, i-1) / akm1k
				denom = akm1*ak - one
				for j = 1; j <= (*nrhs); j++ {
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
		err = goblas.Ztrsm(Left, Upper, Trans, Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        P * B  [ P * (U**T \ (D \ (U \P**T * B) )) ]
		//
		//        Interchange rows K and IPIV(K) of matrix B in reverse order
		//        from the formation order of IPIV(I) vector for Upper case.
		//
		//        (We can do the simple loop over IPIV with increment 1,
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = 1; k <= (*n); k++ {
			kp = absint((*ipiv)[k-1])
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
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
		for k = 1; k <= (*n); k++ {
			kp = absint((*ipiv)[k-1])
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}
		}

		//        Compute (L \P**T * B) -> B    [ (L \P**T * B) ]
		err = goblas.Ztrsm(Left, Lower, NoTrans, Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Compute D \ B -> B   [ D \ (L \P**T * B) ]
		i = 1
		for i <= (*n) {
			if (*ipiv)[i-1] > 0 {
				goblas.Zscal(*nrhs, one/a.Get(i-1, i-1), b.CVector(i-1, 0), *ldb)
			} else if i < (*n) {
				akm1k = e.Get(i - 1)
				akm1 = a.Get(i-1, i-1) / akm1k
				ak = a.Get(i+1-1, i+1-1) / akm1k
				denom = akm1*ak - one
				for j = 1; j <= (*nrhs); j++ {
					bkm1 = b.Get(i-1, j-1) / akm1k
					bk = b.Get(i+1-1, j-1) / akm1k
					b.Set(i-1, j-1, (ak*bkm1-bk)/denom)
					b.Set(i+1-1, j-1, (akm1*bk-bkm1)/denom)
				}
				i = i + 1
			}
			i = i + 1
		}

		//        Compute (L**T \ B) -> B   [ L**T \ (D \ (L \P**T * B) ) ]
		err = goblas.Ztrsm(Left, Lower, Trans, Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        P * B  [ P * (L**T \ (D \ (L \P**T * B) )) ]
		//
		//        Interchange rows K and IPIV(K) of matrix B in reverse order
		//        from the formation order of IPIV(I) vector for Lower case.
		//
		//        (We can do the simple loop over IPIV with decrement -1,
		//        since the ABS value of IPIV(I) represents the row index
		//        of the interchange with row i in both 1x1 and 2x2 pivot cases)
		for k = (*n); k >= 1; k-- {
			kp = absint((*ipiv)[k-1])
			if kp != k {
				goblas.Zswap(*nrhs, b.CVector(k-1, 0), *ldb, b.CVector(kp-1, 0), *ldb)
			}
		}

		//        END Lower
	}
}
