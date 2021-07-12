package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Drqt02 tests DORGRQ, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the RQ factorization of an m-by-n matrix A, DRQT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// rows of A; it compares R(m-k+1:m,n-m+1:n) with
// A(m-k+1:m,1:n)*Q(n-m+1:n,1:n)', and checks that the rows of Q are
// orthonormal.
func Drqt02(m, n, k *int, a, af, q, r *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info int
	var err error
	_ = err

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Copy the last k rows of the factorization to the array Q
	golapack.Dlaset('F', m, n, &rogue, &rogue, q, lda)
	if (*k) < (*n) {
		golapack.Dlacpy('F', k, toPtr((*n)-(*k)), af.Off((*m)-(*k), 0), lda, q.Off((*m)-(*k), 0), lda)
	}
	if (*k) > 1 {
		golapack.Dlacpy('L', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k)+2-1, (*n)-(*k)), lda, q.Off((*m)-(*k)+2-1, (*n)-(*k)), lda)
	}

	//     Generate the last n rows of the matrix Q
	*srnamt = "DORGRQ"
	golapack.Dorgrq(m, n, k, q, lda, tau.Off((*m)-(*k)), work, lwork, &info)

	//     Copy R(m-k+1:m,n-m+1:n)
	golapack.Dlaset('F', k, m, &zero, &zero, r.Off((*m)-(*k), (*n)-(*m)), lda)
	golapack.Dlacpy('U', k, k, af.Off((*m)-(*k), (*n)-(*k)), lda, r.Off((*m)-(*k), (*n)-(*k)), lda)

	//     Compute R(m-k+1:m,n-m+1:n) - A(m-k+1:m,1:n) * Q(n-m+1:n,1:n)'
	err = goblas.Dgemm(mat.NoTrans, mat.Trans, *k, *m, *n, -one, a.Off((*m)-(*k), 0), q, one, r.Off((*m)-(*k), (*n)-(*m)))

	//     Compute norm( R - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', k, n, a.Off((*m)-(*k), 0), lda, rwork)
	resid = golapack.Dlange('1', k, m, r.Off((*m)-(*k), (*n)-(*m)), lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset('F', m, m, &zero, &one, r, lda)
	err = goblas.Dsyrk(mat.Upper, mat.NoTrans, *m, *n, -one, q, one, r)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', 'U', m, r, lda, rwork)

	result.Set(1, (resid/float64(max(1, *n)))/eps)
}
