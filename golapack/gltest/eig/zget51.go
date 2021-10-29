package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget51 generally checks a decomposition of the form
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
func zget51(itype, n int, a, b, u, v *mat.CMatrix, work *mat.CVector, rwork *mat.Vector) (result float64) {
	var cone, czero complex128
	var anorm, one, ten, ulp, unfl, wnorm float64
	var jcol, jdiag, jrow int
	var err error

	one = 1.0
	ten = 10.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	if n <= 0 {
		return
	}

	//     Constants
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     Some Error Checks
	if itype < 1 || itype > 3 {
		result = ten / ulp
		return
	}

	if itype <= 2 {
		//        Tests scaled by the norm(A)
		anorm = math.Max(golapack.Zlange('1', n, n, a, rwork), unfl)

		if itype == 1 {
			//           ITYPE=1: Compute W = A - U B V**H
			golapack.Zlacpy(Full, n, n, a, work.CMatrix(n, opts))
			if err = goblas.Zgemm(NoTrans, NoTrans, n, n, n, cone, u, b, czero, work.CMatrixOff(pow(n, 2), n, opts)); err != nil {
				panic(err)
			}

			if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, -cone, work.CMatrixOff(pow(n, 2), n, opts), v, cone, work.CMatrix(n, opts)); err != nil {
				panic(err)
			}

		} else {
			//           ITYPE=2: Compute W = A - B
			golapack.Zlacpy(Full, n, n, b, work.CMatrix(n, opts))

			for jcol = 1; jcol <= n; jcol++ {
				for jrow = 1; jrow <= n; jrow++ {
					work.Set(jrow+n*(jcol-1)-1, work.Get(jrow+n*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			}
		}

		//        Compute norm(W)/ ( ulp*norm(A) )
		wnorm = golapack.Zlange('1', n, n, work.CMatrix(n, opts), rwork)

		if anorm > wnorm {
			result = (wnorm / anorm) / (float64(n) * ulp)
		} else {
			if anorm < one {
				result = (math.Min(wnorm, float64(n)*anorm) / anorm) / (float64(n) * ulp)
			} else {
				result = math.Min(wnorm/anorm, float64(n)) / (float64(n) * ulp)
			}
		}

	} else {
		//        Tests not scaled by norm(A)
		//
		//        ITYPE=3: Compute  U U**H - I
		if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, cone, u, u, czero, work.CMatrix(n, opts)); err != nil {
			panic(err)
		}

		for jdiag = 1; jdiag <= n; jdiag++ {
			work.Set((n+1)*(jdiag-1), work.Get((n+1)*(jdiag-1))-cone)
		}

		result = math.Min(golapack.Zlange('1', n, n, work.CMatrix(n, opts), rwork), float64(n)) / (float64(n) * ulp)
	}

	return
}
