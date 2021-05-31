package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt02 tests ZUNGQR, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QR factorization of an m-by-n matrix A, ZQRT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// columns of A; it compares R(1:n,1:k) with Q(1:m,1:n)'*A(1:m,1:k),
// and checks that the columns of Q are orthonormal.
func Zqrt02(m, n, k *int, a, af, q, r *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var info int

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)

	srnamt := &gltest.Common.Srnamc.Srnamt

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Zlaset('F', m, n, &rogue, &rogue, q, lda)
	golapack.Zlacpy('L', toPtr((*m)-1), k, af.Off(1, 0), lda, q.Off(1, 0), lda)

	//     Generate the first n columns of the matrix Q
	*srnamt = "ZUNGQR"
	golapack.Zungqr(m, n, k, q, lda, tau, work, lwork, &info)

	//     Copy R(1:n,1:k)
	golapack.Zlaset('F', n, k, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), r, lda)
	golapack.Zlacpy('U', n, k, af, lda, r, lda)

	//     Compute R(1:n,1:k) - Q(1:m,1:n)' * A(1:m,1:k)
	goblas.Zgemm(ConjTrans, NoTrans, n, k, m, toPtrc128(complex(-one, 0)), q, lda, a, lda, toPtrc128(complex(one, 0)), r, lda)

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, k, a, lda, rwork)
	resid = golapack.Zlange('1', n, k, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset('F', n, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), r, lda)
	goblas.Zherk(Upper, ConjTrans, n, m, toPtrf64(-one), q, lda, &one, r, lda)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', 'U', n, r, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *m)))/eps)
}
