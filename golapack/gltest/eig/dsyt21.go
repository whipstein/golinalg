package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dsyt21 generally checks a decomposition of the form
//
//    A = U S U**T
//
// where **T means transpose, A is symmetric, U is orthogonal, and S is
// diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
//
// If ITYPE=1, then U is represented as a dense matrix; otherwise U is
// expressed as a product of Householder transformations, whose vectors
// are stored in the array "V" and whose scaling constants are in "TAU".
// We shall use the letter "V" to refer to the product of Householder
// transformations (which should be equal to U).
//
// Specifically, if ITYPE=1, then:
//
//    RESULT(1) = | A - U S U**T | / ( |A| n ulp ) and
//    RESULT(2) = | I - U U**T | / ( n ulp )
//
// If ITYPE=2, then:
//
//    RESULT(1) = | A - V S V**T | / ( |A| n ulp )
//
// If ITYPE=3, then:
//
//    RESULT(1) = | I - V U**T | / ( n ulp )
//
// For ITYPE > 1, the transformation U is expressed as a product
// V = H(1)...H(n-2),  where H(j) = I  -  tau(j) v(j) v(j)**T and each
// vector v(j) has its first j elements 0 and the remaining n-j elements
// stored in V(j+1:n,j).
func dsyt21(itype int, uplo mat.MatUplo, n, kband int, a *mat.Matrix, d, e *mat.Vector, u, v *mat.Matrix, tau, work, result *mat.Vector) {
	var lower bool
	var cuplo mat.MatUplo
	var anorm, one, ten, ulp, unfl, vsave, wnorm, zero float64
	var iinfo, j, jcol, jr, jrow int
	var err error

	zero = 0.0
	one = 1.0
	ten = 10.0

	result.Set(0, zero)
	if itype == 1 {
		result.Set(1, zero)
	}
	if n <= 0 {
		return
	}

	if uplo == Upper {
		lower = false
		cuplo = Upper
	} else {
		lower = true
		cuplo = Lower
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     Some Error Checks
	if itype < 1 || itype > 3 {
		result.Set(0, ten/ulp)
		return
	}

	//     Do Test 1
	//
	//     Norm of A:
	if itype == 3 {
		anorm = one
	} else {
		anorm = math.Max(golapack.Dlansy('1', cuplo, n, a, work), unfl)
	}

	//     Compute error matrix:
	if itype == 1 {
		//        ITYPE=1: error = A - U S U**T
		golapack.Dlaset(Full, n, n, zero, zero, work.Matrix(n, opts))
		golapack.Dlacpy(cuplo, n, n, a, work.Matrix(n, opts))

		for j = 1; j <= n; j++ {
			err = goblas.Dsyr(cuplo, n, -d.Get(j-1), u.Vector(0, j-1, 1), work.Matrix(n, opts))
		}

		if n > 1 && kband == 1 {
			for j = 1; j <= n-1; j++ {
				if err = goblas.Dsyr2(cuplo, n, -e.Get(j-1), u.Vector(0, j-1, 1), u.Vector(0, j, 1), work.Matrix(n, opts)); err != nil {
					panic(err)
				}
			}
		}
		wnorm = golapack.Dlansy('1', cuplo, n, work.Matrix(n, opts), work.Off(pow(n, 2)))

	} else if itype == 2 {
		//        ITYPE=2: error = V S V**T - A
		golapack.Dlaset(Full, n, n, zero, zero, work.Matrix(n, opts))

		if lower {
			work.Set(pow(n, 2)-1, d.Get(n-1))
			for j = n - 1; j >= 1; j-- {
				if kband == 1 {
					work.Set((n+1)*(j-1)+2-1, (one-tau.Get(j-1))*e.Get(j-1))
					for jr = j + 2; jr <= n; jr++ {
						work.Set((j-1)*n+jr-1, -tau.Get(j-1)*e.Get(j-1)*v.Get(jr-1, j-1))
					}
				}

				vsave = v.Get(j, j-1)
				v.Set(j, j-1, one)
				dlarfy(Lower, n-j, v.Vector(j, j-1, 1), tau.Get(j-1), work.MatrixOff((n+1)*j, n, opts), work.Off(pow(n, 2)))
				v.Set(j, j-1, vsave)
				work.Set((n+1)*(j-1), d.Get(j-1))
			}
		} else {
			work.Set(0, d.Get(0))
			for j = 1; j <= n-1; j++ {
				if kband == 1 {
					work.Set((n+1)*j-1, (one-tau.Get(j-1))*e.Get(j-1))
					for jr = 1; jr <= j-1; jr++ {
						work.Set(j*n+jr-1, -tau.Get(j-1)*e.Get(j-1)*v.Get(jr-1, j))
					}
				}

				vsave = v.Get(j-1, j)
				v.Set(j-1, j, one)
				dlarfy(Upper, j, v.Vector(0, j, 1), tau.Get(j-1), work.Matrix(n, opts), work.Off(pow(n, 2)))
				v.Set(j-1, j, vsave)
				work.Set((n+1)*j, d.Get(j))
			}
		}

		for jcol = 1; jcol <= n; jcol++ {
			if lower {
				for jrow = jcol; jrow <= n; jrow++ {
					work.Set(jrow+n*(jcol-1)-1, work.Get(jrow+n*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			} else {
				for jrow = 1; jrow <= jcol; jrow++ {
					work.Set(jrow+n*(jcol-1)-1, work.Get(jrow+n*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			}
		}
		wnorm = golapack.Dlansy('1', cuplo, n, work.Matrix(n, opts), work.Off(pow(n, 2)))

	} else if itype == 3 {
		//        ITYPE=3: error = U V**T - I
		if n < 2 {
			return
		}
		golapack.Dlacpy(Full, n, n, u, work.Matrix(n, opts))
		if lower {
			if err = golapack.Dorm2r(Right, Trans, n, n-1, n-1, v.Off(1, 0), tau, work.MatrixOff(n, n, opts), work.Off(pow(n, 2))); err != nil {
				panic(err)
			}
		} else {
			if err = golapack.Dorm2l(Right, Trans, n, n-1, n-1, v.Off(0, 1), tau, work.Matrix(n, opts), work.Off(pow(n, 2))); err != nil {
				panic(err)
			}
		}
		if iinfo != 0 {
			result.Set(0, ten/ulp)
			return
		}

		for j = 1; j <= n; j++ {
			work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-one)
		}

		wnorm = golapack.Dlange('1', n, n, work.Matrix(n, opts), work.Off(pow(n, 2)))
	}

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(n)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (math.Min(wnorm, float64(n)*anorm)/anorm)/(float64(n)*ulp))
		} else {
			result.Set(0, math.Min(wnorm/anorm, float64(n))/(float64(n)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U U**T - I
	if itype == 1 {
		if err = goblas.Dgemm(NoTrans, ConjTrans, n, n, n, one, u, u, zero, work.Matrix(n, opts)); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-one)
		}

		result.Set(1, math.Min(golapack.Dlange('1', n, n, work.Matrix(n, opts), work.Off(pow(n, 2))), float64(n))/(float64(n)*ulp))
	}
}
