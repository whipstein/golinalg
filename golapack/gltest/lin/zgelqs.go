package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgelqs Compute a minimum-norm solution
//     min || A*X - B ||
// using the LQ factorization
//     A = L*Q
// computed by ZGELQF.
func Zgelqs(m, n, nrhs *int, a *mat.CMatrix, lda *int, tau *mat.CVector, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var cone, czero complex128
	var err error
	_ = err

	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

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
		gltest.Xerbla([]byte("ZGELQS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     Solve L*X = B(1:m,:)
	err = goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, *m, *nrhs, cone, a, b)

	//     Set B(m+1:n,:) to zero
	if (*m) < (*n) {
		golapack.Zlaset('F', toPtr((*n)-(*m)), nrhs, &czero, &czero, b.Off((*m), 0), ldb)
	}

	//     B := Q' * B
	golapack.Zunmlq('L', 'C', n, nrhs, m, a, lda, tau, b, ldb, work, lwork, info)
}
