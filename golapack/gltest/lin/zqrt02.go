package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// zqrt02 tests Zungqr, which generates an m-by-n matrix Q with
// orthonornmal columns that is defined as the product of k elementary
// reflectors.
//
// Given the QR factorization of an m-by-n matrix A, ZQRT02 generates
// the orthogonal matrix Q defined by the factorization of the first k
// columns of A; it compares R(1:n,1:k) with Q(1:m,1:n)'*A(1:m,1:k),
// and checks that the columns of Q are orthonormal.
func zqrt02(m, n, k int, a, af, q, r *mat.CMatrix, tau, work *mat.CVector, lwork int, rwork, result *mat.Vector) {
	var rogue complex128
	var anorm, eps, one, resid, zero float64
	var err error

	zero = 0.0
	one = 1.0
	rogue = (-1.0e+10 + (-1.0e+10)*1i)

	srnamt := &gltest.Common.Srnamc.Srnamt

	eps = golapack.Dlamch(Epsilon)

	//     Copy the first k columns of the factorization to the array Q
	golapack.Zlaset(Full, m, n, rogue, rogue, q)
	golapack.Zlacpy(Lower, m-1, k, af.Off(1, 0), q.Off(1, 0))

	//     Generate the first n columns of the matrix Q
	*srnamt = "Zungqr"
	if err = golapack.Zungqr(m, n, k, q, tau, work, lwork); err != nil {
		panic(err)
	}

	//     Copy R(1:n,1:k)
	golapack.Zlaset(Full, n, k, complex(zero, 0), complex(zero, 0), r)
	golapack.Zlacpy(Upper, n, k, af, r)

	//     Compute R(1:n,1:k) - Q(1:m,1:n)' * A(1:m,1:k)
	if err = goblas.Zgemm(ConjTrans, NoTrans, n, k, m, complex(-one, 0), q, a, complex(one, 0), r); err != nil {
		panic(err)
	}

	//     Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
	anorm = golapack.Zlange('1', m, k, a, rwork)
	resid = golapack.Zlange('1', n, k, r, rwork)
	if anorm > zero {
		result.Set(0, ((resid/float64(max(1, m)))/anorm)/eps)
	} else {
		result.Set(0, zero)
	}

	//     Compute I - Q'*Q
	golapack.Zlaset(Full, n, n, complex(zero, 0), complex(one, 0), r)
	if err = goblas.Zherk(Upper, ConjTrans, n, m, -one, q, one, r); err != nil {
		panic(err)
	}

	//     Compute norm( I - Q'*Q ) / ( M * EPS ) .
	resid = golapack.Zlansy('1', Upper, n, r, rwork)

	result.Set(1, (resid/float64(max(1, m)))/eps)
}
