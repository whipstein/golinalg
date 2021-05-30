package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// DsytrsAa2stage solves a system of linear equations A*X = B with a real
// symmetric matrix A using the factorization A = U**T*T*U or
// A = L*T*L**T computed by DSYTRF_AA_2STAGE.
func DsytrsAa2stage(uplo byte, n *int, nrhs *int, a *mat.Matrix, lda *int, tb *mat.Vector, ltb *int, ipiv *[]int, ipiv2 *[]int, b *mat.Matrix, ldb *int, info *int) {
	var upper bool
	var one float64
	var ldtb, nb int

	one = 1.0

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
		gltest.Xerbla([]byte("DSYTRS_AA_2STAGE"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	//     Read NB and compute LDTB
	nb = int(tb.Get(0))
	ldtb = (*ltb) / (*n)

	if upper {
		//        Solve A*X = B, where A = U**T*T*U.
		if (*n) > nb {
			//           Pivot, P**T * B -> B
			Dlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, func() *int { y := 1; return &y }())

			//           Compute (U**T \ B) -> B    [ (U**T \P**T * B) ]
			goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.Unit, toPtr((*n)-nb), nrhs, &one, a.Off(0, nb+1-1), lda, b.Off(nb+1-1, 0), ldb)

		}

		//        Compute T \ B -> B   [ T \ (U**T \P**T * B) ]
		Dgbtrs('N', n, &nb, &nb, nrhs, tb.Matrix(ldtb, opts), &ldtb, ipiv2, b, ldb, info)
		if (*n) > nb {
			//           Compute (U \ B) -> B   [ U \ (T \ (U**T \P**T * B) ) ]
			goblas.Dtrsm(mat.Left, mat.Upper, mat.NoTrans, mat.Unit, toPtr((*n)-nb), nrhs, &one, a.Off(0, nb+1-1), lda, b.Off(nb+1-1, 0), ldb)

			//           Pivot, P * B -> B  [ P * (U \ (T \ (U**T \P**T * B) )) ]
			Dlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, toPtr(-1))

		}

	} else {
		//        Solve A*X = B, where A = L*T*L**T.
		if (*n) > nb {
			//           Pivot, P**T * B -> B
			Dlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, func() *int { y := 1; return &y }())

			//           Compute (L \ B) -> B    [ (L \P**T * B) ]
			goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, toPtr((*n)-nb), nrhs, &one, a.Off(nb+1-1, 0), lda, b.Off(nb+1-1, 0), ldb)

		}

		//        Compute T \ B -> B   [ T \ (L \P**T * B) ]
		Dgbtrs('N', n, &nb, &nb, nrhs, tb.Matrix(ldtb, opts), &ldtb, ipiv2, b, ldb, info)
		if (*n) > nb {
			//           Compute (L**T \ B) -> B   [ L**T \ (T \ (L \P**T * B) ) ]
			goblas.Dtrsm(mat.Left, mat.Lower, mat.Trans, mat.Unit, toPtr((*n)-nb), nrhs, &one, a.Off(nb+1-1, 0), lda, b.Off(nb+1-1, 0), ldb)

			//           Pivot, P * B -> B  [ P * (L**T \ (T \ (L \P**T * B) )) ]
			Dlaswp(nrhs, b, ldb, toPtr(nb+1), n, ipiv, toPtr(-1))

		}
	}
}
