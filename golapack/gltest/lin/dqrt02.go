package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Dqrt02 tests DORGQR, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QR factorization of an m-by-n matrix A, DQRT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// columns of A; it compares R(1:n,1:k) with Q(1:m,1:n)'*A(1:m,1:k),
// and checks that the columns of Q are orthonormal.
func Dqrt02(m, n, k *int, a, af, q, r *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info int
	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Dlaset('F', m, n, &rogue, &rogue, q, lda)
	golapack.Dlacpy('L', toPtr((*m)-1), k, af.Off(1, 0), lda, q.Off(1, 0), lda)

	//     Generate the first n columns of the matrix Q
	*srnamt = "DORGQR"
	golapack.Dorgqr(m, n, k, q, lda, tau, work, lwork, &info)

	//     Copy R(1:n,1:k)
	golapack.Dlaset('F', n, k, &zero, &zero, r, lda)
	golapack.Dlacpy('U', n, k, af, lda, r, lda)

	//     Compute R(1:n,1:k) - Q(1:m,1:n)' * A(1:m,1:k)
	goblas.Dgemm(mat.Trans, mat.NoTrans, n, k, m, toPtrf64(-one), q, lda, a, lda, &one, r, lda)

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, k, a, lda, rwork)
	resid = golapack.Dlange('1', n, k, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset('F', n, n, &zero, &one, r, lda)
	goblas.Dsyrk(mat.Upper, mat.Trans, n, m, toPtrf64(-one), q, lda, &one, r, lda)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', 'U', n, r, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *m)))/eps)
}
