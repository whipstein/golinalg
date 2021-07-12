package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqrt01 tests ZGEQRF, which computes the QR factorization of an m-by-n
// matrix A, and partially tests ZUNGQR which forms the m-by-m
// orthogonal matrix Q.
//
// ZQRT01 compares R with Q'*A, and checks that Q is orthogonal.
func Zqrt01(m, n *int, a, af, q, r *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
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
	*srnamt = "ZGEQRF"
	golapack.Zgeqrf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Zlaset('F', m, m, &rogue, &rogue, q, lda)
	golapack.Zlacpy('L', toPtr((*m)-1), n, af.Off(1, 0), lda, q.Off(1, 0), lda)

	//     Generate the m-by-m matrix Q
	*srnamt = "ZUNGQR"
	golapack.Zungqr(m, m, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy R
	golapack.Zlaset('F', m, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), r, lda)
	golapack.Zlacpy('U', m, n, af, lda, r, lda)

	//     Compute R - Q'*A
	err = goblas.Zgemm(ConjTrans, NoTrans, *m, *n, *m, complex(-one, 0), q, a, complex(one, 0), r)

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)
	resid = golapack.Zlange('1', m, n, r, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset('F', m, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), r, lda)
	err = goblas.Zherk(Upper, ConjTrans, *m, *m, -one, q, one, r)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', 'U', m, r, lda, rwork)

	result.Set(1, (resid/float64(max(1, *m)))/eps)
}
