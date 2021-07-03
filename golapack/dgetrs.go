package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgetrs solves a system of linear equations
//    A * X = B  or  A**T * X = B
// with a general N-by-N matrix A using the LU factorization computed
// by DGETRF.
func Dgetrs(trans byte, n *int, nrhs *int, a *mat.Matrix, lda *int, ipiv *[]int, b *mat.Matrix, ldb *int, info *int) {
	var notran bool
	var err error
	_ = err

	one := 1.0

	//     Test the input parameters.
	(*info) = 0
	notran = trans == 'N'
	if !notran && trans != 'T' && trans != 'C' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *n) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *n) {
		(*info) = -8
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGETRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 {
		return
	}

	if notran {
		//        Solve A * X = B.
		//
		//        Apply row interchanges to the right hand sides.
		Dlaswp(nrhs, b, ldb, func() *int { y := 1; return &y }(), n, ipiv, func() *int { y := 1; return &y }())

		//        Solve L*X = B, overwriting B with X.
		err = goblas.Dtrsm(mat.Left, mat.Lower, mat.NoTrans, mat.Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Solve U*X = B, overwriting B with X.
		err = goblas.Dtrsm(mat.Left, mat.Upper, mat.NoTrans, mat.NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)
	} else {
		//        Solve A**T * X = B.
		//
		//        Solve U**T *X = B, overwriting B with X.
		err = goblas.Dtrsm(mat.Left, mat.Upper, mat.Trans, mat.NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Solve L**T *X = B, overwriting B with X.
		err = goblas.Dtrsm(mat.Left, mat.Lower, mat.Trans, mat.Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Apply row interchanges to the solution vectors.
		Dlaswp(nrhs, b, ldb, func() *int { y := 1; return &y }(), n, ipiv, toPtr(-1))
	}
}
