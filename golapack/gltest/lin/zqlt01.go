package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zqlt01 tests ZGEQLF, which computes the QL factorization of an m-by-n
// matrix A, and partially tests ZUNGQL which forms the m-by-m
// orthogonal matrix Q.
//
// ZQLT01 compares L with Q'*A, and checks that Q is orthogonal.
func Zqlt01(m, n *int, a, af, q, l *mat.CMatrix, lda *int, tau, work *mat.CVector, lwork *int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var info, minmn int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	minmn = minint(*m, *n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy('F', m, n, a, lda, af, lda)

	//     Factorize the matrix A in the array AF.
	*srnamt = "ZGEQLF"
	golapack.Zgeqlf(m, n, af, lda, tau, work, lwork, &info)

	//     Copy details of Q
	golapack.Zlaset('F', m, m, &rogue, &rogue, q, lda)
	if (*m) >= (*n) {
		if (*n) < (*m) && (*n) > 0 {
			golapack.Zlacpy('F', toPtr((*m)-(*n)), n, af, lda, q.Off(0, (*m)-(*n)+1-1), lda)
		}
		if (*n) > 1 {
			golapack.Zlacpy('U', toPtr((*n)-1), toPtr((*n)-1), af.Off((*m)-(*n)+1-1, 1), lda, q.Off((*m)-(*n)+1-1, (*m)-(*n)+2-1), lda)
		}
	} else {
		if (*m) > 1 {
			golapack.Zlacpy('U', toPtr((*m)-1), toPtr((*m)-1), af.Off(0, (*n)-(*m)+2-1), lda, q.Off(0, 1), lda)
		}
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "ZUNGQL"
	golapack.Zungql(m, m, &minmn, q, lda, tau, work, lwork, &info)

	//     Copy L
	golapack.Zlaset('F', m, n, toPtrc128(complex(zero, 0)), toPtrc128(complex(zero, 0)), l, lda)
	if (*m) >= (*n) {
		if (*n) > 0 {
			golapack.Zlacpy('L', n, n, af.Off((*m)-(*n)+1-1, 0), lda, l.Off((*m)-(*n)+1-1, 0), lda)
		}
	} else {
		if (*n) > (*m) && (*m) > 0 {
			golapack.Zlacpy('F', m, toPtr((*n)-(*m)), af, lda, l, lda)
		}
		if (*m) > 0 {
			golapack.Zlacpy('L', m, m, af.Off(0, (*n)-(*m)+1-1), lda, l.Off(0, (*n)-(*m)+1-1), lda)
		}
	}

	//     Compute L - Q'*A
	err = goblas.Zgemm(ConjTrans, NoTrans, *m, *n, *m, complex(-one, 0), q, *lda, a, *lda, complex(one, 0), l, *lda)

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, lda, rwork)
	resid = golapack.Zlange('1', m, n, l, lda, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(maxint(1, *m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset('F', m, m, toPtrc128(complex(zero, 0)), toPtrc128(complex(one, 0)), l, lda)
	err = goblas.Zherk(Upper, ConjTrans, *m, *m, -one, q, *lda, one, l, *lda)

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', 'U', m, l, lda, rwork)

	result.Set(1, (resid/float64(maxint(1, *m)))/eps)
}
