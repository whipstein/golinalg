package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zrqt01 tests Zgerqf, which computes the RQ factorization of an m-by-n
// matrix A, and partially tests Zungrq which forms the n-by-n
// orthogonal matrix Q.
//
// ZRQT01 compares R with A*Q', and checks that Q is orthogonal.
func zrqt01(m, n int, a, af, q, r *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
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
	*srnamt = "Zgerqf"
	if err = golapack.Zgerqf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Zlaset(Full, n, n, rogue, rogue, q)
	if m <= n {
		if m > 0 && m < n {
			golapack.Zlacpy(Full, m, n-m, af, q.Off(n-m, 0))
		}
		if m > 1 {
			golapack.Zlacpy(Lower, m-1, m-1, af.Off(1, n-m), q.Off(n-m+2-1, n-m))
		}
	} else {
		if n > 1 {
			golapack.Zlacpy(Lower, n-1, n-1, af.Off(m-n+2-1, 0), q.Off(1, 0))
		}
	}

	//     Generate the n-by-n matrix Q
	*srnamt = "Zungrq"
	if err = golapack.Zungrq(n, n, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R
	golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), r)
	if m <= n {
		if m > 0 {
			golapack.Zlacpy(Upper, m, m, af.Off(0, n-m), r.Off(0, n-m))
		}
	} else {
		if m > n && n > 0 {
			golapack.Zlacpy(Full, m-n, n, af, r)
		}
		if n > 0 {
			golapack.Zlacpy(Upper, n, n, af.Off(m-n, 0), r.Off(m-n, 0))
		}
	}

	//     Compute R - A*Q'
	if err = goblas.Zgemm(NoTrans, ConjTrans, m, n, n, complex(-one, 0), a, q, complex(one, 0), r); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), r)
	if err = goblas.Zherk(Upper, NoTrans, n, n, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', Upper, n, r, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
