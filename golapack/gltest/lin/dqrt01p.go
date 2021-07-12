package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqrt01p tests DGEQRFP, which computes the QR factorization of an m-by-n
// matrix A, and partially tests DORGQR which forms the m-by-m
// orthogonal matrix Q.
//
// DQRT01P compares R with Q'*A, and checks that Q is orthogonal.
func Dqrt01p(m, n *int, a, af, q, r *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info, minmn int
	var err error
	_ = err

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	minmn = min(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "DGEQRFP"
	golapack.Dgeqrfp(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Dlaset('F', m, m, &rogue, &rogue, q, lda)
	golapack.Dlacpy('L', toPtr((*m)-1), n, af.Off(1, 0), lda, q.Off(1, 0), lda)

	//     Generate the m-by-m matrix Q
	*srnamt = "DORGQR"
	golapack.Dorgqr(m, m, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy R
	golapack.Dlaset('F', m, n, &zero, &zero, r, lda)
	golapack.Dlacpy('U', m, n, af, lda, r, lda)

	//     Compute R - Q'*A
	err = goblas.Dgemm(mat.Trans, mat.NoTrans, *m, *n, *m, -one, q, a, one, r)

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, lda, rwork)
	resid = golapack.Dlange('1', m, n, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset('F', m, m, &zero, &one, r, lda)
	err = goblas.Dsyrk(mat.Upper, mat.Trans, *m, *m, -one, q, one, r)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', 'U', m, r, lda, rwork)

	result.Set(1, (resid/float64(max(1, *m)))/eps)
}
