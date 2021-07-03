package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgeqrs Solve the least squares problem
//     min || A*X - B ||
// using the QR factorization
//     A = Q*R
// computed by DGEQRF.
func Dgeqrs(m, n, nrhs *int, a *mat.Matrix, lda *int, tau *mat.Vector, b *mat.Matrix, ldb *int, work *mat.Vector, lwork, info *int) {
	var one float64
	var err error
	_ = err

	one = 1.0

	//     Test the input arguments.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*n) > (*m) {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < maxint(1, *m) {
		(*info) = -5
	} else if (*ldb) < maxint(1, *m) {
		(*info) = -8
	} else if (*lwork) < 1 || (*lwork) < (*nrhs) && (*m) > 0 && (*n) > 0 {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGEQRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     B := Q' * B
	golapack.Dormqr('L', 'T', m, nrhs, n, a, lda, tau, b, ldb, work, lwork, info)

	//     Solve R*X = B(1:n,:)
	err = goblas.Dtrsm(mat.Left, mat.Upper, mat.NoTrans, mat.NonUnit, *n, *nrhs, one, a, *lda, b, *ldb)
}
