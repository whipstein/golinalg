package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgeqls Solve the least squares problem
//     minint || A*X - B ||
// using the QL factorization
//     A = Q*L
// computed by ZGEQLF.
func Zgeqls(m, n, nrhs *int, a *mat.CMatrix, lda *int, tau *mat.CVector, b *mat.CMatrix, ldb *int, work *mat.CVector, lwork, info *int) {
	var one complex128
	var err error
	_ = err

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
		gltest.Xerbla([]byte("ZGEQLS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 || (*nrhs) == 0 || (*m) == 0 {
		return
	}

	//     B := Q' * B
	golapack.Zunmql('L', 'C', m, nrhs, n, a, lda, tau, b, ldb, work, lwork, info)

	//     Solve L*X = B(m-n+1:m,:)
	err = goblas.Ztrsm(Left, Lower, NoTrans, NonUnit, *n, *nrhs, one, a.Off((*m)-(*n)+1-1, 0), *lda, b.Off((*m)-(*n)+1-1, 0), *ldb)
}
