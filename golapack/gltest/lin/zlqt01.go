package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlqt01 tests ZGELQF, which computes the LQ factorization of an m-by-n
// matrix A, and partially tests ZUNGLQ which forms the n-by-n
// orthogonal matrix Q.
//
// ZLQT01 compares L with A*Q', and checks that Q is orthogonal.
func Zlqt01(m, n *int, a, af, q, l *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var info, minmn int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	minmn = min(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "ZGELQF"
	golapack.Zgelqf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Zlaset('F', n, n, &rogue, &rogue, q, lda)
	if (*n) > 1 {
		golapack.Zlacpy('U', m, toPtr((*n)-1), af.Off(0, 1), lda, q.Off(0, 1), lda)
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "ZUNGLQ"
	golapack.Zunglq(n, n, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy L
	golapack.Zlaset('F', m, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), l, lda)
	golapack.Zlacpy('L', m, n, af, lda, l, lda)

	//     Compute L - A*Q'
	err = goblas.Zgemm(NoTrans, ConjTrans, *m, *n, *n, complex(-one, 0), a, q, complex(one, 0), l)

	//     Compute norm( L - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)
	resid = golapack.Zlange('1', m, n, l, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset('F', n, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), l, lda)
	err = goblas.Zherk(Upper, NoTrans, *n, *n, -one, q, one, l)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', 'U', n, l, lda, rwork)

	result.Set(1, (resid/float64(max(1, *n)))/eps)
}
