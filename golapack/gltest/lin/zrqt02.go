package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zrqt02 tests Zungrq, which generates an m-by-n matrix Q with
// orthonornmal rows that is defined as the product of k elementary
// reflectors.
//
// Given the RQ factorization of an m-by-n matrix A, ZRQT02 generates
// the orthogonal matrix Q defined by the factorization of the last k
// rows of A; it compares R(m-k+1:m,n-m+1:n) with
// A(m-k+1:m,1:n)*Q(n-m+1:n,1:n)', and checks that the rows of Q are
// orthonormal.
func zrqt02(m, n, k int, a, af, q, r *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
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

	//     Copy the last k rows of the factorization to the array Q
	golapack.Zlaset(Full, m, n, rogue, rogue, q)
	if k < n {
		golapack.Zlacpy(Full, k, n-k, af.Off(m-k, 0), q.Off(m-k, 0))
	}
	if k > 1 {
		golapack.Zlacpy(Lower, k-1, k-1, af.Off(m-k+2-1, n-k), q.Off(m-k+2-1, n-k))
	}

	//     Generate the last n rows of the matrix Q
	*srnamt = "Zungrq"
	if err = golapack.Zungrq(m, n, k, q, tau.Off(m-k), work, lwork); err != nil {
		panic(err)
	}

	//     Copy R(m-k+1:m,n-m+1:n)
	golapack.Zlaset(Full, k, m, complex(zero, 0), complex(zero, 0), r.Off(m-k, n-m))
	golapack.Zlacpy(Upper, k, k, af.Off(m-k, n-k), r.Off(m-k, n-k))

	//     Compute R(m-k+1:m,n-m+1:n) - A(m-k+1:m,1:n) * Q(n-m+1:n,1:n)'
	if err = goblas.Zgemm(NoTrans, ConjTrans, k, m, n, complex(-one, 0), a.Off(m-k, 0), q, complex(one, 0), r.Off(m-k, n-m)); err != nil {
		panic(err)
	}

	//     Compute norm( R - A*Q' ) / ( N * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', k, n, a.Off(m-k, 0), rwork)
	resid = golapack.Zlange('1', k, m, r.Off(m-k, n-m), rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, n)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q*Q'
	golapack.Zlaset(Full, m, m, complex(zero, 0), complex(one, 0), r)
	if err = goblas.Zherk(Upper, NoTrans, m, n, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q*Q' ) / ( N * EPS ) .
	resid = golapack.Zlansy('1', Upper, m, r, rwork)

	result.Set(1, (resid/float64(max(1, n)))/eps)
}
