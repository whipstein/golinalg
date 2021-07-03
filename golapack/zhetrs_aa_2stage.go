package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhetrsaa2stage solves a system of linear equations A*X = B with a
// hermitian matrix A using the factorization A = U**H*T*U or
// A = L*T*L**H computed by ZHETRF_AA_2STAGE.
func Zhetrsaa2stage(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, tb *mat.CVector, ltb *int, ipiv, ipiv2 *[]int, b *mat.CMatrix, ldb, info *int) {
	var upper bool
	var one complex128
	var ldtb, nb int
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
	} else if (*ltb) < (4 * (*n)) {
		(*info) = -7
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHETRS_AA_2STAGE"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	//     Read NB and compute LDTB
	nb = int(tb.GetRe(0))
	ldtb = (*ltb) / (*n)

	if upper {
		//        Solve A*X = B, where A = U**H*T*U.
		if (*n) > nb {
			//           Pivot, P**T * B -> B
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, func() *int { y := 1; return &y }())

			//           Compute (U**H \ B) -> B    [ (U**H \P**T * B) ]
			err = goblas.Ztrsm(Left, Upper, ConjTrans, Unit, (*n)-nb, *nrhs, one, a.Off(0, nb+1-1), *lda, b.Off(nb+1-1, 0), *ldb)

		}

		//        Compute T \ B -> B   [ T \ (U**H \P**T * B) ]
		Zgbtrs('N', n, &nb, &nb, nrhs, tb.CMatrix(ldtb, opts), &ldtb, ipiv2, b, ldb, info)
		if (*n) > nb {
			//           Compute (U \ B) -> B   [ U \ (T \ (U**H \P**T * B) ) ]
			err = goblas.Ztrsm(Left, Upper, NoTrans, Unit, (*n)-nb, *nrhs, one, a.Off(0, nb+1-1), *lda, b.Off(nb+1-1, 0), *ldb)

			//           Pivot, P * B -> B  [ P * (U \ (T \ (U**H \P**T * B) )) ]
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, toPtr(-1))

		}

	} else {
		//        Solve A*X = B, where A = L*T*L**H.
		if (*n) > nb {
			//           Pivot, P**T * B -> B
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, func() *int { y := 1; return &y }())

			//           Compute (L \ B) -> B    [ (L \P**T * B) ]
			err = goblas.Ztrsm(Left, Lower, NoTrans, Unit, (*n)-nb, *nrhs, one, a.Off(nb+1-1, 0), *lda, b.Off(nb+1-1, 0), *ldb)

		}

		//        Compute T \ B -> B   [ T \ (L \P**T * B) ]
		Zgbtrs('N', n, &nb, &nb, nrhs, tb.CMatrix(ldtb, opts), &ldtb, ipiv2, b, ldb, info)
		if (*n) > nb {
			//           Compute (L**H \ B) -> B   [ L**H \ (T \ (L \P**T * B) ) ]
			err = goblas.Ztrsm(Left, Lower, ConjTrans, Unit, (*n)-nb, *nrhs, one, a.Off(nb+1-1, 0), *lda, b.Off(nb+1-1, 0), *ldb)

			//           Pivot, P * B -> B  [ P * (L**H \ (T \ (L \P**T * B) )) ]
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, toPtr(-1))

		}
	}
}
