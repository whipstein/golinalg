package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// drqt01 tests Dgerqf, which computes the RQ factorization of an m-by-n
// matrix A, and partially tests Dorgrq which forms the n-by-n
// orthogonal matrix Q.
//
// DRQT01 compares R with A*Q', and checks that Q is orthogonal.
func drqt01(m, n int, a, af, q, r *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
	var anorm, eps, one, resid, rogue, zero float64
	var minmn int
	var err error

	srnamt := &gltest.Common.Srnamc.Srnamt

	zero = 0.0
	one = 1.0
	rogue = -1.0e+10

	minmn = min(m, n)
	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix A to the array AF.
	golapack.Dlacpy(Full, m, n, a, af)

	//     Factorize the matrix A in the array AF.
	*srnamt = "Dgerqf"
	if err = golapack.Dgerqf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Dlaset(Full, n, n, rogue, rogue, q)
	if m <= n {
		if m > 0 && m < n {
			golapack.Dlacpy(Full, m, n-m, af, q.Off(n-m, 0))
		}
		if m > 1 {
			golapack.Dlacpy(Lower, m-1, m-1, af.Off(1, n-m), q.Off(n-m+2-1, n-m))
		}
	} else {
		if n > 1 {
			golapack.Dlacpy(Lower, n-1, n-1, af.Off(m-n+2-1, 0), q.Off(1, 0))
		}
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "Dorgrq"
	if err = golapack.Dorgrq(n, n, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Dlaset(Full, m, n, zero, zero, r)
	if m <= n {
		if m > 0 {
			golapack.Dlacpy(Upper, m, m, af.Off(0, n-m), r.Off(0, n-m))
		}
	} else {
		if m > n && n > 0 {
			golapack.Dlacpy(Full, m-n, n, af, r)
		}
		if n > 0 {
			golapack.Dlacpy(Upper, n, n, af.Off(m-n, 0), r.Off(m-n, 0))
		}
	}

	//     Compute R - A*Q'
	if err = goblas.Dgemm(NoTrans, Trans, m, n, n, -one, a, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, rwork)
	resid = golapack.Dlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Dlaset(Full, n, n, zero, one, r)
	if err = goblas.Dsyrk(Upper, NoTrans, n, n, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Dlansy('1', Upper, n, r, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
