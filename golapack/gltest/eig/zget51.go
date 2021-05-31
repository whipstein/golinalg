package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zget51 generally checks a decomposition of the form
//
//              A = U B V**H
//
//      where **H means conjugate transpose and U and V are unitary.
//
//      Specifically, if ITYPE=1
//
//              RESULT = | A - U B V**H | / ( |A| n ulp )
//
//      If ITYPE=2, then:
//
//              RESULT = | A - B | / ( |A| n ulp )
//
//      If ITYPE=3, then:
//
//              RESULT = | I - U U**H | / ( n ulp )
func Zget51(itype, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, work *mat.CVector, rwork *mat.Vector, result *float64) {
	var cone, czero complex128
	var anorm, one, ten, ulp, unfl, wnorm, zero float64
	var jcol, jdiag, jrow int

	zero = 0.0
	one = 1.0
	ten = 10.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	(*result) = zero
	if (*n) <= 0 {
		return
	}

	//     Constants
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     Some Error Checks
	if (*itype) < 1 || (*itype) > 3 {
		(*result) = ten / ulp
		return
	}

	if (*itype) <= 2 {
		//        Tests scaled by the norm(A)
		anorm = maxf64(golapack.Zlange('1', n, n, a, lda, rwork), unfl)

		if (*itype) == 1 {
			//           ITYPE=1: Compute W = A - U B V**H
			golapack.Zlacpy(' ', n, n, a, lda, work.CMatrix(*n, opts), n)
			goblas.Zgemm(NoTrans, NoTrans, n, n, n, &cone, u, ldu, b, ldb, &czero, work.CMatrixOff(powint(*n, 2)+1-1, *n, opts), n)

			goblas.Zgemm(NoTrans, ConjTrans, n, n, n, toPtrc128(-cone), work.CMatrixOff(powint(*n, 2)+1-1, *n, opts), n, v, ldv, &cone, work.CMatrix(*n, opts), n)

		} else {
			//           ITYPE=2: Compute W = A - B
			golapack.Zlacpy(' ', n, n, b, ldb, work.CMatrix(*n, opts), n)

			for jcol = 1; jcol <= (*n); jcol++ {
				for jrow = 1; jrow <= (*n); jrow++ {
					work.Set(jrow+(*n)*(jcol-1)-1, work.Get(jrow+(*n)*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			}
		}

		//        Compute norm(W)/ ( ulp*norm(A) )
		wnorm = golapack.Zlange('1', n, n, work.CMatrix(*n, opts), n, rwork)

		if anorm > wnorm {
			(*result) = (wnorm / anorm) / (float64(*n) * ulp)
		} else {
			if anorm < one {
				(*result) = (minf64(wnorm, float64(*n)*anorm) / anorm) / (float64(*n) * ulp)
			} else {
				(*result) = minf64(wnorm/anorm, float64(*n)) / (float64(*n) * ulp)
			}
		}

	} else {
		//        Tests not scaled by norm(A)
		//
		//        ITYPE=3: Compute  U U**H - I
		goblas.Zgemm(NoTrans, ConjTrans, n, n, n, &cone, u, ldu, u, ldu, &czero, work.CMatrix(*n, opts), n)

		for jdiag = 1; jdiag <= (*n); jdiag++ {
			work.Set(((*n)+1)*(jdiag-1)+1-1, work.Get(((*n)+1)*(jdiag-1)+1-1)-cone)
		}

		(*result) = minf64(golapack.Zlange('1', n, n, work.CMatrix(*n, opts), n, rwork), float64(*n)) / (float64(*n) * ulp)
	}
}
