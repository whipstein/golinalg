package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlqt02 tests ZUNGLQ, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the LQ factorization of an m-by-n matrix A, ZLQT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// rows of A; it compares L(1:k,1:m) with A(1:k,1:n)*Q(1:m,1:n)', and
// checks that the rows of Q are orthonormal.
func Zlqt02(m, n, k *int, a, af, q, l *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var info int

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k rows of the factorization to the array Q
	golapack.Zlaset('F', m, n, &rogue, &rogue, q, lda)
	golapack.Zlacpy('U', k, toPtr((*n)-1), af.Off(0, 1), lda, q.Off(0, 1), lda)

	//     Generate the first n columns of the matrix Q
	*srnamt = "ZUNGLQ"
	golapack.Zunglq(m, n, k, q, lda, tau, work, lwork, &info)

	//     Copy L(1:k,1:m)
	golapack.Zlaset('F', k, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), l, lda)
	golapack.Zlacpy('L', k, m, af, lda, l, lda)

	//     Compute L(1:k,1:m) - A(1:k,1:n) * Q(1:m,1:n)'
	goblas.Zgemm(NoTrans, ConjTrans, k, m, n, toPtrc128(complex(-one, 0)), a, lda, q, lda, toPtrc128(complex(one, 0)), l, lda)

	//     Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', k, n, a, lda, rwork)
	resid = golapack.Zlange('1', k, m, l, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset('F', m, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), l, lda)
	goblas.Zherk(Upper, NoTrans, m, n, toPtrf64(-one), q, lda, &one, l, lda)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', 'U', m, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *n)))/eps)
}
