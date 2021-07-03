package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dqlt02 tests DORGQL, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QL factorization of an m-by-n matrix A, DQLT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// columns of A; it compares L(m-n+1:m,n-k+1:n) with
// Q(1:m,m-n+1:m)'*A(1:m,n-k+1:n), and checks that the columns of Q are
// orthonormal.
func Dqlt02(m, n, k *int, a, af, q, l *mat.Matrix, lda *int, tau, work *mat.Vector, lwork *int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var info int
	var err error
	_ = err

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Copy the last k columns of the factorization to the array Q
	golapack.Dlaset('F', m, n, &rogue, &rogue, q, lda)
	if (*k) < (*m) {
		golapack.Dlacpy('F', toPtr((*m)-(*k)), k, af.Off(0, (*n)-(*k)+1-1), lda, q.Off(0, (*n)-(*k)+1-1), lda)
	}
	if (*k) > 1 {
		golapack.Dlacpy('U', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k)+1-1, (*n)-(*k)+2-1), lda, q.Off((*m)-(*k)+1-1, (*n)-(*k)+2-1), lda)
	}

	//     Generate the last n columns of the matrix Q
	*srnamt = "DORGQL"
	golapack.Dorgql(m, n, k, q, lda, tau.Off((*n)-(*k)+1-1), work, lwork, &info)

	//     Copy L(m-n+1:m,n-k+1:n)
	golapack.Dlaset('F', n, k, &zero, &zero, l.Off((*m)-(*n)+1-1, (*n)-(*k)+1-1), lda)
	golapack.Dlacpy('L', k, k, af.Off((*m)-(*k)+1-1, (*n)-(*k)+1-1), lda, l.Off((*m)-(*k)+1-1, (*n)-(*k)+1-1), lda)

	//     Compute L(m-n+1:m,n-k+1:n) - Q(1:m,m-n+1:m)' * A(1:m,n-k+1:n)
	err = goblas.Dgemm(mat.Trans, mat.NoTrans, *n, *k, *m, -one, q, *lda, a.Off(0, (*n)-(*k)+1-1), *lda, one, l.Off((*m)-(*n)+1-1, (*n)-(*k)+1-1), *lda)

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, k, a.Off(0, (*n)-(*k)+1-1), lda, rwork)
	resid = golapack.Dlange('1', n, k, l.Off((*m)-(*n)+1-1, (*n)-(*k)+1-1), lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset('F', n, n, &zero, &one, l, lda)
	err = goblas.Dsyrk(mat.Upper, mat.Trans, *n, *m, -one, q, *lda, one, l, *lda)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', 'U', n, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *m)))/eps)
}
