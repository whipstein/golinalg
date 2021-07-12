package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zsytrsaa2stage solves a system of linear equations A*X = B with a complex
// symmetric matrix A using the factorization A = U**T*T*U or
// A = L*T*L**T computed by ZSYTRF_AA_2STAGE.
func Zsytrsaa2stage(uplo byte, n, nrhs *int, a *mat.CMatrix, lda *int, tb *mat.CVector, ltb *int, ipiv, ipiv2 *[]int, b *mat.CMatrix, ldb, info *int) {
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
	} else if (*lda) < max(1, *n) {
		(*info) = -5
	} else if (*ltb) < (4 * (*n)) {
		(*info) = -7
	} else if (*ldb) < max(1, *n) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZSYTRS_AA_2STAGE"), -(*info))
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
		//        Solve A*X = B, where A = U**T*T*U.
		if (*n) > nb {
			//           Pivot, P**T * B -> B
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, func() *int { y := 1; return &y }())

			//           Compute (U**T \ B) -> B    [ (U**T \P**T * B) ]
			err = goblas.Ztrsm(Left, Upper, Trans, Unit, (*n)-nb, *nrhs, one, a.Off(0, nb), b.Off(nb, 0))

		}

		//        Compute T \ B -> B   [ T \ (U**T \P**T * B) ]
		Zgbtrs('N', n, &nb, &nb, nrhs, tb.CMatrix(ldtb, opts), &ldtb, ipiv2, b, ldb, info)
		if (*n) > nb {
			//           Compute (U \ B) -> B   [ U \ (T \ (U**T \P**T * B) ) ]
			err = goblas.Ztrsm(Left, Upper, NoTrans, Unit, (*n)-nb, *nrhs, one, a.Off(0, nb), b.Off(nb, 0))

			//           Pivot, P * B -> B  [ P * (U \ (T \ (U**T \P**T * B) )) ]
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, toPtr(-1))

		}

	} else {
		//        Solve A*X = B, where A = L*T*L**T.
		if (*n) > nb {
			//           Pivot, P**T * B -> B
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, func() *int { y := 1; return &y }())

			//           Compute (L \ B) -> B    [ (L \P**T * B) ]
			err = goblas.Ztrsm(Left, Lower, NoTrans, Unit, (*n)-nb, *nrhs, one, a.Off(nb, 0), b.Off(nb, 0))

		}

		//        Compute T \ B -> B   [ T \ (L \P**T * B) ]
		Zgbtrs('N', n, &nb, &nb, nrhs, tb.CMatrix(ldtb, opts), &ldtb, ipiv2, b, ldb, info)
		if (*n) > nb {
			//           Compute (L**T \ B) -> B   [ L**T \ (T \ (L \P**T * B) ) ]
			err = goblas.Ztrsm(Left, Lower, Trans, Unit, (*n)-nb, *nrhs, one, a.Off(nb, 0), b.Off(nb, 0))

			//           Pivot, P * B -> B  [ P * (L**T \ (T \ (L \P**T * B) )) ]
			Zlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, toPtr(-1))

		}
	}
}
