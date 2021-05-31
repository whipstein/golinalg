package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zrzt01 returns
//      || A - R*Q || / ( M * eps * ||A|| )
// for an upper trapezoidal A that was factored with ZTZRZF.
func Zrzt01(m, n *int, a, af *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int) (zrzt01Return float64) {
	var norma, one, zero float64
	var i, info, j int

	rwork := vf(1)

	zero = 0.0
	one = 1.0

	zrzt01Return = zero

	if (*lwork) < (*m)*(*n)+(*m) {
		gltest.Xerbla([]byte("ZRZT01"), 8)
		return
	}

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	norma = golapack.Zlange('O', m, n, a, lda, rwork)

	//     Copy upper triangle R
	golapack.Zlaset('F', m, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), work.CMatrix(*m, opts), m)
	for j = 1; j <= (*m); j++ {
		for i = 1; i <= j; i++ {
			work.Set((j-1)*(*m)+i-1, af.Get(i-1, j-1))
		}
	}

	//     R = R * P(1) * ... *P(m)
	golapack.Zunmrz('R', 'N', m, n, m, toPtr((*n)-(*m)), af, lda, tau, work.CMatrix(*m, opts), m, work.Off((*m)*(*n)+1-1), toPtr((*lwork)-(*m)*(*n)), &info)

	//     R = R - A
	for i = 1; i <= (*n); i++ {
		goblas.Zaxpy(m, toPtrc128(complex(-one, 0)), a.CVector(0, i-1), func() *int { y := 1; return &y }(), work.Off((i-1)*(*m)+1-1), func() *int { y := 1; return &y }())
	}

	zrzt01Return = golapack.Zlange('O', m, n, work.CMatrix(*m, opts), m, rwork)

	zrzt01Return = zrzt01Return / (golapack.Dlamch(Epsilon) * float64(maxint(*m, *n)))
	if norma != zero {
		zrzt01Return = zrzt01Return / norma
	}

	return
}
