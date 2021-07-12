package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgelqs Compute a minimum-norm solution
//     min || A*X - B ||
// using the LQ factorization
//     A = L*Q
// computed by DGELQF.
func Dgelqs(m, n, nrhs *int, a *mat.Matrix, lda *int, tau *mat.Vector, b *mat.Matrix, ldb *int, work *mat.Vector, lwork, info *int) {
	var one, zero float64
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	//     Test the input parameters.
	(*info) = 0
	if (*m) < 0 {
		(*info) = -1
	} else if (*n) < 0 || (*m) > (*n) {
		(*info) = -2
	} else if (*nrhs) < 0 {
		(*info) = -3
	} else if (*lda) < max(1, *m) {
		(*info) = -5
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	} else if (*lwork) < 1 || (*lwork) < (*nrhs) && (*m) > 0 && (*n) > 0 {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGELQS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     Solve L*X = B(1:m,:)
	err = goblas.Dtrsm(Left, Lower, NoTrans, NonUnit, *m, *nrhs, one, a, b)

	//     Set B(m+1:n,:) to zero
	if (*m) < (*n) {
		golapack.Dlaset('F', toPtr((*n)-(*m)), nrhs, &zero, &zero, b.Off((*m), 0), ldb)
	}

	//     B := Q' * B
	golapack.Dormlq('L', 'T', n, nrhs, m, a, lda, tau, b, ldb, work, lwork, info)
}
