package golapack

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgetrs solves a system of linear equations
//    A * X = B,  A**T * X = B,  or  A**H * X = B
// with a general N-by-N matrix A using the LU factorization computed
// by ZGETRF.
func Zgetrs(trans byte, n, nrhs *int, a *mat.CMatrix, lda *int, ipiv *[]int, b *mat.CMatrix, ldb, info *int) {
	var notran bool
	var one complex128
	var err error
	_ = err

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZGETRS"), -(*info))
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
		Zlaswp(nrhs, b, ldb, func() *int { y := 1; return &y }(), n, ipiv, func() *int { y := 1; return &y }())

		//        Solve L*X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Lower, NoTrans, Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Solve U*X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)
	} else {
		//        Solve A**T * X = B  or A**H * X = B.
		//
		//        Solve U**T *X = B or U**H *X = B, overwriting B with X.
		err = goblas.Ztrsm(Left, Upper, mat.TransByte(trans), NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Solve L**T *X = B, or L**H *X = B overwriting B with X.
		err = goblas.Ztrsm(Left, Lower, mat.TransByte(trans), Unit, *n, *nrhs, one, a, *lda, b, *ldb)

		//        Apply row interchanges to the solution vectors.
		Zlaswp(nrhs, b, ldb, func() *int { y := 1; return &y }(), n, ipiv, toPtr(-1))
	}
}
