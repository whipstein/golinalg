package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt01p tests Zgeqrfp, which computes the QR factorization of an m-by-n
// matrix A, and partially tests Zungqr which forms the m-by-m
// orthogonal matrix Q.
//
// ZQRT01P compares R with Q'*A, and checks that Q is orthogonal.
func zqrt01p(m, n int, a, af, q, r *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var minmn int
	var err error

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)

	srnamt := &gltest.Common.Srnamc.Srnamt

	minmn = min(m, n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Zlacpy(Full, m, n, a, af)

	//     Factorize the matrix A in the array AF.
	*srnamt = "Zgeqrfp"
	if err = golapack.Zgeqrfp(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Zlaset(Full, m, m, rogue, rogue, q)
	golapack.Zlacpy(Lower, m-1, n, af.Off(1, 0), q.Off(1, 0))

	//     Generate the m-by-m matrix Q
	*srnamt = "Zungqr"
	if err = golapack.Zungqr(m, m, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), r)
	golapack.Zlacpy(Upper, m, n, af, r)

	//     Compute R - Q'*A
	if err = goblas.Zgemm(ConjTrans, NoTrans, m, n, m, complex(-one, 0), q, a, complex(one, 0), r); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset(Full, m, m, complex(zero, 0), complex(one, 0), r)
	if err = goblas.Zherk(Upper, ConjTrans, m, m, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', Upper, m, r, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
