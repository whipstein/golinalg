package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dgerqs Compute a minimum-norm solution
//     min || A*X - B ||
// using the RQ factorization
//     A = R*Q
// computed by DGERQF.
func Dgerqs(m, n, nrhs *int, a *mat.Matrix, lda *int, tau *mat.Vector, b *mat.Matrix, ldb *int, work *mat.Vector, lwork, info *int) {
	var one, zero float64

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
	} else if (*lda) < maxint(1, (*m)) {
		(*info) = -5
	} else if (*ldb) < maxint(1, (*n)) {
		(*info) = -8
	} else if (*lwork) < 1 || (*lwork) < (*nrhs) && (*m) > 0 && (*n) > 0 {
		(*info) = -10
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGERQS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     Solve R*X = B((*n)-(*m)+1:(*n),:)
	goblas.Dtrsm(Left, Upper, NoTrans, NonUnit, m, nrhs, &one, a.Off(0, (*n)-(*m)+1-1), lda, b.Off((*n)-(*m)+1-1, 0), ldb)

	//     Set B(1:(*n)-(*m),:) to zero
	golapack.Dlaset('F', toPtr((*n)-(*m)), &(*nrhs), &zero, &zero, b, &(*ldb))

	//     B := Q' * B
	golapack.Dormrq('L', 'T', n, nrhs, m, a, lda, tau, b, ldb, work, lwork, info)
}