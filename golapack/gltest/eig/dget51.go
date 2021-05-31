package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dget51 generally checks a decomposition of the form
//
//              A = U B V'
//
//      where ' means transpose and U and V are orthogonal.
//
//      Specifically, if ITYPE=1
//
//              RESULT = | A - U B V' | / ( |A| n ulp )
//
//      If ITYPE=2, then:
//
//              RESULT = | A - B | / ( |A| n ulp )
//
//      If ITYPE=3, then:
//
//              RESULT = | I - UU' | / ( n ulp )
func Dget51(itype, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, work *mat.Vector, result *float64) {
	var anorm, one, ten, ulp, unfl, wnorm, zero float64
	var jcol, jdiag, jrow int

	zero = 0.0
	one = 1.0
	ten = 10.0

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
		anorm = maxf64(golapack.Dlange('1', n, n, a, lda, work), unfl)

		if (*itype) == 1 {
			//           ITYPE=1: Compute W = A - UBV'
			golapack.Dlacpy(' ', n, n, a, lda, work.Matrix(*n, opts), n)
			goblas.Dgemm(NoTrans, NoTrans, n, n, n, &one, u, ldu, b, ldb, &zero, work.MatrixOff(int(math.Pow(float64(*n), 2))+1-1, *n, opts), n)

			goblas.Dgemm(NoTrans, ConjTrans, n, n, n, toPtrf64(-one), work.MatrixOff(int(math.Pow(float64(*n), 2))+1-1, *n, opts), n, v, ldv, &one, work.Matrix(*n, opts), n)

		} else {
			//           ITYPE=2: Compute W = A - B
			golapack.Dlacpy(' ', n, n, b, ldb, work.Matrix(*n, opts), n)

			for jcol = 1; jcol <= (*n); jcol++ {
				for jrow = 1; jrow <= (*n); jrow++ {
					work.Set(jrow+(*n)*(jcol-1)-1, work.Get(jrow+(*n)*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			}
		}

		//        Compute norm(W)/ ( ulp*norm(A) )
		wnorm = golapack.Dlange('1', n, n, work.Matrix(*n, opts), n, work.Off(int(math.Pow(float64(*n), 2))+1-1))

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
		//        ITYPE=3: Compute  UU' - I
		goblas.Dgemm(NoTrans, ConjTrans, n, n, n, &one, u, ldu, u, ldu, &zero, work.Matrix(*n, opts), n)

		for jdiag = 1; jdiag <= (*n); jdiag++ {
			work.Set(((*n)+1)*(jdiag-1)+1-1, work.Get(((*n)+1)*(jdiag-1)+1-1)-one)
		}

		(*result) = minf64(golapack.Dlange('1', n, n, work.Matrix(*n, opts), n, work.Off(int(math.Pow(float64(*n), 2))+1-1)), float64(*n)) / (float64(*n) * ulp)
	}
}
