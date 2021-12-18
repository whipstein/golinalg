package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dget51 generally checks a decomposition of the form
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
func dget51(itype, n int, a, b, u, v *mat.Matrix, work *mat.Vector) (result float64) {
	var anorm, one, ten, ulp, unfl, wnorm, zero float64
	var jcol, jdiag, jrow int
	var err error

	zero = 0.0
	one = 1.0
	ten = 10.0

	result = zero
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
		anorm = math.Max(golapack.Dlange('1', n, n, a, work), unfl)

		if itype == 1 {
			//           ITYPE=1: Compute W = A - UBV'
			golapack.Dlacpy(Full, n, n, a, work.Matrix(n, opts))
			if err = work.Off(pow(n, 2)).Matrix(n, opts).Gemm(NoTrans, NoTrans, n, n, n, one, u, b, zero); err != nil {
				panic(err)
			}

			if err = work.Matrix(n, opts).Gemm(NoTrans, ConjTrans, n, n, n, -one, work.Off(pow(n, 2)).Matrix(n, opts), v, one); err != nil {
				panic(err)
			}

		} else {
			//           ITYPE=2: Compute W = A - B
			golapack.Dlacpy(Full, n, n, b, work.Matrix(n, opts))

			for jcol = 1; jcol <= n; jcol++ {
				for jrow = 1; jrow <= n; jrow++ {
					work.Set(jrow+n*(jcol-1)-1, work.Get(jrow+n*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			}
		}

		//        Compute norm(W)/ ( ulp*norm(A) )
		wnorm = golapack.Dlange('1', n, n, work.Matrix(n, opts), work.Off(pow(n, 2)))

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
		//        ITYPE=3: Compute  UU' - I
		if err = work.Matrix(n, opts).Gemm(NoTrans, ConjTrans, n, n, n, one, u, u, zero); err != nil {
			panic(err)
		}

		for jdiag = 1; jdiag <= n; jdiag++ {
			work.Set((n+1)*(jdiag-1), work.Get((n+1)*(jdiag-1))-one)
		}

		result = math.Min(golapack.Dlange('1', n, n, work.Matrix(n, opts), work.Off(pow(n, 2))), float64(n)) / (float64(n) * ulp)
	}

	return
}
