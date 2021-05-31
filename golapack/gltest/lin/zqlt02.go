package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqlt02 tests ZUNGQL, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QL factorization of an m-by-n matrix A, ZQLT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// columns of A; it compares L(m-n+1:m,n-k+1:n) with
// Q(1:m,m-n+1:m)'*A(1:m,n-k+1:n), and checks that the columns of Q are
// orthonormal.
func Zqlt02(m, n, k *int, a, af, q, l *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var info int

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Quick return if possible
	if (*m) == 0 || (*n) == 0 || (*k) == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Copy the last k columns of the factorization to the array Q
	golapack.Zlaset('F', m, n, &rogue, &rogue, q, lda)
	if (*k) < (*m) {
		golapack.Zlacpy('F', toPtr((*m)-(*k)), k, af.Off(0, (*n)-(*k)+1-1), lda, q.Off(0, (*n)-(*k)+1-1), lda)
	}
	if (*k) > 1 {
		golapack.Zlacpy('U', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k)+1-1, (*n)-(*k)+2-1), lda, q.Off((*m)-(*k)+1-1, (*n)-(*k)+2-1), lda)
	}

	//     Generate the last n columns of the matrix Q
	*srnamt = "ZUNGQL"
	golapack.Zungql(m, n, k, q, lda, tau.Off((*n)-(*k)+1-1), work, lwork, &info)

	//     Copy L(m-n+1:m,n-k+1:n)
	golapack.Zlaset('F', n, k, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), l.Off((*m)-(*n)+1-1, (*n)-(*k)+1-1), lda)
	golapack.Zlacpy('L', k, k, af.Off((*m)-(*k)+1-1, (*n)-(*k)+1-1), lda, l.Off((*m)-(*k)+1-1, (*n)-(*k)+1-1), lda)

	//     Compute L(m-n+1:m,n-k+1:n) - Q(1:m,m-n+1:m)' * A(1:m,n-k+1:n)
	goblas.Zgemm(ConjTrans, NoTrans, n, k, m, toPtrc128(complex(-one, 0)), q, lda, a.Off(0, (*n)-(*k)+1-1), lda, toPtrc128(complex(one, 0)), l.Off((*m)-(*n)+1-1, (*n)-(*k)+1-1), lda)

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, k, a.Off(0, (*n)-(*k)+1-1), lda, rwork)
	resid = golapack.Zlange('1', n, k, l.Off((*m)-(*n)+1-1, (*n)-(*k)+1-1), lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset('F', n, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), l, lda)
	goblas.Zherk(Upper, ConjTrans, n, m, toPtrf64(-one), q, lda, &one, l, lda)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', 'U', n, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *m)))/eps)
}
