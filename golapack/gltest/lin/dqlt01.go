package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// dqlt01 tests DGEQLF, which computes the QL factorization of an m-by-n
// matrix A, and partially tests DORGQL which forms the m-by-m
// orthogonal matrix Q.
//
// DQLT01 compares L with Q'*A, and checks that Q is orthogonal.
func dqlt01(m, n int, a, af, q, l *mat.Matrix, tau, work *mat.Vector, lwork int, rwork, result *mat.Vector) {
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
	*srnamt = "Dgeqlf"
	if err = golapack.Dgeqlf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Dlaset(Full, m, m, rogue, rogue, q)
	if m >= n {
		if n < m && n > 0 {
			golapack.Dlacpy(Full, m-n, n, af, q.Off(0, m-n))
		}
		if n > 1 {
			golapack.Dlacpy(Upper, n-1, n-1, af.Off(m-n, 1), q.Off(m-n, m-n+2-1))
		}
	} else {
		if m > 1 {
			golapack.Dlacpy(Upper, m-1, m-1, af.Off(0, n-m+2-1), q.Off(0, 1))
		}
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "Dorgql"
	if err = golapack.Dorgql(m, m, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Dlaset(Full, m, n, zero, zero, l)
	if m >= n {
		if n > 0 {
			golapack.Dlacpy(Lower, n, n, af.Off(m-n, 0), l.Off(m-n, 0))
		}
	} else {
		if n > m && m > 0 {
			golapack.Dlacpy(Full, m, n-m, af, l)
		}
		if m > 0 {
			golapack.Dlacpy(Lower, m, m, af.Off(0, n-m), l.Off(0, n-m))
		}
	}

	//     Compute L - Q'*A
	if err = goblas.Dgemm(mat.Trans, mat.NoTrans, m, n, m, -one, q, a, one, l); err != nil {
		panic(err)
	}

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Dlange('1', m, n, a, rwork)
	resid = golapack.Dlange('1', m, n, l, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Dlaset(Full, m, m, zero, one, l)
	if err = goblas.Dsyrk(Upper, Trans, m, m, -one, q, one, l); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Dlansy('1', Upper, m, l, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
