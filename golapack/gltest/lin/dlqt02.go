package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dlqt02 tests DORGLQ, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the LQ factorization of an m-by-n matrix A, DLQT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// rows of A; it compares L(1:k,1:m) with A(1:k,1:n)*Q(1:m,1:n)', and
// checks that the rows of Q are orthonormal.
func Dlqt02(m, n, k *int, a, af, q, l *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info int
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k rows of the factorization to the array Q
	golapack.Dlaset('F', m, n, &rogue, &rogue, q, lda)
	golapack.Dlacpy('U', k, toPtr((*n)-1), af.Off(0, 1), lda, q.Off(0, 1), lda)

	//     Generate the first n columns of the matrix Q
	*srnamt = "DORGLQ"
	golapack.Dorglq(m, n, k, q, lda, tau, work, lwork, &info)

	//     Copy L(1:k,1:m)
	golapack.Dlaset('F', k, m, &zero, &zero, l, lda)
	golapack.Dlacpy('L', k, m, af, lda, l, lda)

	//     Compute L(1:k,1:m) - A(1:k,1:n) * Q(1:m,1:n)'
	goblas.Dgemm(mat.NoTrans, mat.Trans, k, m, n, toPtrf64(-one), a, lda, q, lda, &one, l, lda)

	//     Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', k, n, a, lda, rwork)
	resid = golapack.Dlange('1', k, m, l, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset('F', m, m, &zero, &one, l, lda)
	goblas.Dsyrk(Upper, NoTrans, m, n, toPtrf64(-one), q, lda, &one, l, lda)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', 'U', m, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *n)))/eps)
}
