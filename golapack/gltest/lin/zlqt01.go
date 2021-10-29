package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zlqt01 tests Zgelqf, which computes the LQ factorization of an m-by-n
// matrix A, and partially tests Zunglq which forms the n-by-n
// orthogonal matrix Q.
//
// ZLQT01 compares L with A*Q', and checks that Q is orthogonal.
func zlqt01(m, n int, a, af, q, l *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
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
	*srnamt = "Zgelqf"
	if err = golapack.Zgelqf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Zlaset(Full, n, n, rogue, rogue, q)
	if n > 1 {
		golapack.Zlacpy(Upper, m, n-1, af.Off(0, 1), q.Off(0, 1))
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "Zunglq"
	if err = golapack.Zunglq(n, n, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), l)
	golapack.Zlacpy(Lower, m, n, af, l)

	//     Compute L - A*Q'
	if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, complex(-one, 0), a, q, complex(one, 0), l); err != nil {
		panic(err)
	}

	//     Compute norm( L - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, l, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), l)
	if err = goblas.Zherk(Upper, NoTrans, n, n, -one, q, one, l); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', Upper, n, l, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
