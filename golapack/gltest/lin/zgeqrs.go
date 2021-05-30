package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zgeqrs Solve the least squares problem
//     minint || A*X - B ||
// using the QR factorization
//     A = Q*R
// computed by ZGEQRF.
func Zgeqrs(m, n, nrhs *int, a *mat.CMatrix, lda *int, tau *mat.CVector, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var one complex128

	one = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZGEQRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     B := Q' * B
	golapack.Zunmqr('L', 'C', m, nrhs, n, a, lda, tau, b, ldb, work, lwork, info)

	//     Solve R*X = B(1:n,:)
	goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, n, nrhs, &one, a, lda, b, ldb)
}
