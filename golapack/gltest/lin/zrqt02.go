package lin

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zrqt02 tests ZUNGRQ, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the RQ factorization of an m-by-n matrix A, ZRQT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// rows of A; it compares R(m-k+1:m,n-m+1:n) with
// A(m-k+1:m,1:n)*Q(n-m+1:n,1:n)', and checks that the rows of Q are
// orthonormal.
func Zrqt02(m, n, k *int, a, af, q, r *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
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

	//     Copy the last k rows of the factorization to the array Q
	golapack.Zlaset('F', m, n, &rogue, &rogue, q, lda)
	if (*k) < (*n) {
		golapack.Zlacpy('F', k, toPtr((*n)-(*k)), af.Off((*m)-(*k)+1-1, 0), lda, q.Off((*m)-(*k)+1-1, 0), lda)
	}
	if (*k) > 1 {
		golapack.Zlacpy('L', toPtr((*k)-1), toPtr((*k)-1), af.Off((*m)-(*k)+2-1, (*n)-(*k)+1-1), lda, q.Off((*m)-(*k)+2-1, (*n)-(*k)+1-1), lda)
	}

	//     Generate the last n rows of the matrix Q
	*srnamt = "ZUNGRQ"
	golapack.Zungrq(m, n, k, q, lda, tau.Off((*m)-(*k)+1-1), work, lwork, &info)

	//     Copy R(m-k+1:m,n-m+1:n)
	golapack.Zlaset('F', k, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), r.Off((*m)-(*k)+1-1, (*n)-(*m)+1-1), lda)
	golapack.Zlacpy('U', k, k, af.Off((*m)-(*k)+1-1, (*n)-(*k)+1-1), lda, r.Off((*m)-(*k)+1-1, (*n)-(*k)+1-1), lda)

	//     Compute R(m-k+1:m,n-m+1:n) - A(m-k+1:m,1:n) * Q(n-m+1:n,1:n)'
	goblas.Zgemm(NoTrans, ConjTrans, k, m, n, toPtrc128(complex(-one, 0)), a.Off((*m)-(*k)+1-1, 0), lda, q, lda, toPtrc128(complex(one, 0)), r.Off((*m)-(*k)+1-1, (*n)-(*m)+1-1), lda)

	//     Compute norm( R - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', k, n, a.Off((*m)-(*k)+1-1, 0), lda, rwork)
	resid = golapack.Zlange('1', k, m, r.Off((*m)-(*k)+1-1, (*n)-(*m)+1-1), lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset('F', m, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), r, lda)
	goblas.Zherk(Upper, NoTrans, m, n, toPtrf64(-one), q, lda, &one, r, lda)

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', 'U', m, r, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *n)))/eps)
}