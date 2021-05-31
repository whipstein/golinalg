package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zrzt02 returns
//      || I - Q'*Q || / ( M * eps)
// where the matrix Q is defined by the Householder transformations
// generated by ZTZRZF.
func Zrzt02(m, n *int, af *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int) (zrzt02Return float64) {
	var one, zero float64
	var i, info int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zrzt02Return = zero

	if (*lwork) < (*n)*(*n)+(*n) {
		gltest.Xerbla([]byte("ZRZT02"), 7)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	//     Q := I
	golapack.Zlaset('F', n, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), work.CMatrix(*n, opts), n)

	//     Q := P(1) * ... * P(m) * Q
	golapack.Zunmrz('L', 'N', n, n, m, toPtr((*n)-(*m)), af, lda, tau, work.CMatrix(*n, opts), n, work.Off((*n)*(*n)+1-1), toPtr((*lwork)-(*n)*(*n)), &info)

	//     Q := P(m)' * ... * P(1)' * Q
	golapack.Zunmrz('L', 'C', n, n, m, toPtr((*n)-(*m)), af, lda, tau, work.CMatrix(*n, opts), n, work.Off((*n)*(*n)+1-1), toPtr((*lwork)-(*n)*(*n)), &info)

	//     Q := Q - I
	for i = 1; i <= (*n); i++ {
		work.Set((i-1)*(*n)+i-1, work.Get((i-1)*(*n)+i-1)-complex(one, 0))
	}

	zrzt02Return = golapack.Zlange('O', n, n, work.CMatrix(*n, opts), n, rwork) / (golapack.Dlamch(Epsilon) * float64(maxint(*m, *n)))
	return
}
