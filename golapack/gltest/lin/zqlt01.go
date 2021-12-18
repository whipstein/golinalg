package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqlt01 tests Zgeqlf, which computes the QL factorization of an m-by-n
// matrix A, and partially tests Zungql which forms the m-by-m
// orthogonal matrix Q.
//
// ZQLT01 compares L with Q'*A, and checks that Q is orthogonal.
func zqlt01(m, n int, a, af, q, l *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
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
	*srnamt = "Zgeqlf"
	if err = golapack.Zgeqlf(m, n, af, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy details of Q
	golapack.Zlaset(Full, m, m, rogue, rogue, q)
	if m >= n {
		if n < m && n > 0 {
			golapack.Zlacpy(Full, m-n, n, af, q.Off(0, m-n))
		}
		if n > 1 {
			golapack.Zlacpy(Upper, n-1, n-1, af.Off(m-n, 1), q.Off(m-n, m-n+2-1))
		}
	} else {
		if m > 1 {
			golapack.Zlacpy(Upper, m-1, m-1, af.Off(0, n-m+2-1), q.Off(0, 1))
		}
	}

	//     Generate the m-by-m matrix Q
	*srnamt = "Zungql"
	if err = golapack.Zungql(m, m, minmn, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy L
	golapack.Zlaset(Full, m, n, complex(zero, 0), complex(zero, 0), l)
	if m >= n {
		if n > 0 {
			golapack.Zlacpy(Lower, n, n, af.Off(m-n, 0), l.Off(m-n, 0))
		}
	} else {
		if n > m && m > 0 {
			golapack.Zlacpy(Full, m, n-m, af, l)
		}
		if m > 0 {
			golapack.Zlacpy(Lower, m, m, af.Off(0, n-m), l.Off(0, n-m))
		}
	}

	//     Compute L - Q'*A
	if err = l.Gemm(ConjTrans, NoTrans, m, n, m, complex(-one, 0), q, a, complex(one, 0)); err != nil {
		panic(err)
	}

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, n, a, rwork)
	resid = golapack.Zlange('1', m, n, l, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset(Full, m, m, complex(zero, 0), complex(one, 0), l)
	if err = l.Herk(Upper, ConjTrans, m, m, -one, q, one); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', Upper, m, l, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
