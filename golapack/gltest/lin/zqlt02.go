package lin

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqlt02 tests Zungql, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QL factorization of an m-by-n matrix A, ZQLT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// columns of A; it compares L(m-n+1:m,n-k+1:n) with
// Q(1:m,m-n+1:m)'*A(1:m,n-k+1:n), and checks that the columns of Q are
// orthonormal.
func zqlt02(m, n, k int, a, af, q, l *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var err error

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)
	srnamt := &gltest.Common.Srnamc.Srnamt

	//     Quick return if possible
	if m == 0 || n == 0 || k == 0 {
		result.Set(0, zero)
		result.Set(1, zero)
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Copy the last k columns of the factorization to the array Q
	golapack.Zlaset(Full, m, n, rogue, rogue, q)
	if k < m {
		golapack.Zlacpy(Full, m-k, k, af.Off(0, n-k), q.Off(0, n-k))
	}
	if k > 1 {
		golapack.Zlacpy(Upper, k-1, k-1, af.Off(m-k, n-k+2-1), q.Off(m-k, n-k+2-1))
	}

	//     Generate the last n columns of the matrix Q
	*srnamt = "Zungql"
	if err = golapack.Zungql(m, n, k, q, tau.Off(n-k), work, lwork); err != nil {
		panic(err)
	}

	//     Copy L(m-n+1:m,n-k+1:n)
	golapack.Zlaset(Full, n, k, complex(zero, 0), complex(zero, 0), l.Off(m-n, n-k))
	golapack.Zlacpy(Lower, k, k, af.Off(m-k, n-k), l.Off(m-k, n-k))

	//     Compute L(m-n+1:m,n-k+1:n) - Q(1:m,m-n+1:m)' * A(1:m,n-k+1:n)
	if err = l.Off(m-n, n-k).Gemm(ConjTrans, NoTrans, n, k, m, complex(-one, 0), q, a.Off(0, n-k), complex(one, 0)); err != nil {
		panic(err)
	}

	//     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, k, a.Off(0, n-k), rwork)
	resid = golapack.Zlange('1', n, k, l.Off(m-n, n-k), rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), l)
	if err = l.Herk(Upper, ConjTrans, n, m, -one, q, one); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', Upper, n, l, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
