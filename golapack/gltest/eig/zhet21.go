package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zhet21 generally checks a decomposition of the form
//
//    A = U S U**H
//
// where **H means conjugate transpose, A is hermitian, U is unitary, and
// S is diagonal (if KBAND=0) or (real) symmetric tridiagonal (if
// KBAND=1).
//
// If ITYPE=1, then U is represented as a dense matrix; otherwise U is
// expressed as a product of Householder transformations, whose vectors
// are stored in the array "V" and whose scaling constants are in "TAU".
// We shall use the letter "V" to refer to the product of Householder
// transformations (which should be equal to U).
//
// Specifically, if ITYPE=1, then:
//
//    RESULT(1) = | A - U S U**H | / ( |A| n ulp ) and
//    RESULT(2) = | I - U U**H | / ( n ulp )
//
// If ITYPE=2, then:
//
//    RESULT(1) = | A - V S V**H | / ( |A| n ulp )
//
// If ITYPE=3, then:
//
//    RESULT(1) = | I - U V**H | / ( n ulp )
//
// For ITYPE > 1, the transformation U is expressed as a product
// V = H(1)...H(n-2),  where H(j) = I  -  tau(j) v(j) v(j)**H and each
// vector v(j) has its first j elements 0 and the remaining n-j elements
// stored in V(j+1:n,j).
func Zhet21(itype *int, uplo byte, n, kband *int, a *mat.CMatrix, lda *int, d, e *mat.Vector, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, tau, work *mat.CVector, rwork, result *mat.Vector) {
	var lower bool
	var cuplo byte
	var cone, czero, vsave complex128
	var anorm, one, ten, ulp, unfl, wnorm, zero float64
	var iinfo, j, jcol, jr, jrow int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	ten = 10.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	result.Set(0, zero)
	if (*itype) == 1 {
		result.Set(1, zero)
	}
	if (*n) <= 0 {
		return
	}

	if uplo == 'U' {
		lower = false
		cuplo = 'U'
	} else {
		lower = true
		cuplo = 'L'
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)

	//     Some Error Checks
	if (*itype) < 1 || (*itype) > 3 {
		result.Set(0, ten/ulp)
		return
	}

	//     Do Test 1
	//
	//     Norm of A:
	if (*itype) == 3 {
		anorm = one
	} else {
		anorm = math.Max(golapack.Zlanhe('1', cuplo, n, a, lda, rwork), unfl)
	}

	//     Compute error matrix:
	if (*itype) == 1 {
		//        ITYPE=1: error = A - U S U**H
		golapack.Zlaset('F', n, n, &czero, &czero, work.CMatrix(*n, opts), n)
		golapack.Zlacpy(cuplo, n, n, a, lda, work.CMatrix(*n, opts), n)

		for j = 1; j <= (*n); j++ {
			err = goblas.Zher(mat.UploByte(cuplo), *n, -d.Get(j-1), u.CVector(0, j-1, 1), work.CMatrix(*n, opts))
		}

		if (*n) > 1 && (*kband) == 1 {
			for j = 1; j <= (*n)-1; j++ {
				err = goblas.Zher2(mat.UploByte(cuplo), *n, -complex(e.Get(j-1), 0), u.CVector(0, j-1, 1), u.CVector(0, j-1-1, 1), work.CMatrix(*n, opts))
			}
		}
		wnorm = golapack.Zlanhe('1', cuplo, n, work.CMatrix(*n, opts), n, rwork)

	} else if (*itype) == 2 {
		//        ITYPE=2: error = V S V**H - A
		golapack.Zlaset('F', n, n, &czero, &czero, work.CMatrix(*n, opts), n)

		if lower {
			work.SetRe(pow(*n, 2)-1, d.Get((*n)-1))
			for j = (*n) - 1; j >= 1; j -= 1 {
				if (*kband) == 1 {
					work.Set(((*n)+1)*(j-1)+2-1, (cone-tau.Get(j-1))*e.GetCmplx(j-1))
					for jr = j + 2; jr <= (*n); jr++ {
						work.Set((j-1)*(*n)+jr-1, -tau.Get(j-1)*e.GetCmplx(j-1)*v.Get(jr-1, j-1))
					}
				}

				vsave = v.Get(j, j-1)
				v.SetRe(j, j-1, one)
				golapack.Zlarfy('L', toPtr((*n)-j), v.CVector(j, j-1), func() *int { y := 1; return &y }(), tau.GetPtr(j-1), work.CMatrixOff(((*n)+1)*j, *n, opts), n, work.Off(pow(*n, 2)))
				v.Set(j, j-1, vsave)
				work.Set(((*n)+1)*(j-1), d.GetCmplx(j-1))
			}
		} else {
			work.Set(0, d.GetCmplx(0))
			for j = 1; j <= (*n)-1; j++ {
				if (*kband) == 1 {
					work.Set(((*n)+1)*j-1, (cone-tau.Get(j-1))*e.GetCmplx(j-1))
					for jr = 1; jr <= j-1; jr++ {
						work.Set(j*(*n)+jr-1, -tau.Get(j-1)*e.GetCmplx(j-1)*v.Get(jr-1, j))
					}
				}

				vsave = v.Get(j-1, j)
				v.SetRe(j-1, j, one)
				golapack.Zlarfy('U', &j, v.CVector(0, j), func() *int { y := 1; return &y }(), tau.GetPtr(j-1), work.CMatrix(*n, opts), n, work.Off(pow(*n, 2)))
				v.Set(j-1, j, vsave)
				work.Set(((*n)+1)*j, d.GetCmplx(j))
			}
		}

		for jcol = 1; jcol <= (*n); jcol++ {
			if lower {
				for jrow = jcol; jrow <= (*n); jrow++ {
					work.Set(jrow+(*n)*(jcol-1)-1, work.Get(jrow+(*n)*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			} else {
				for jrow = 1; jrow <= jcol; jrow++ {
					work.Set(jrow+(*n)*(jcol-1)-1, work.Get(jrow+(*n)*(jcol-1)-1)-a.Get(jrow-1, jcol-1))
				}
			}
		}
		wnorm = golapack.Zlanhe('1', cuplo, n, work.CMatrix(*n, opts), n, rwork)

	} else if (*itype) == 3 {
		//        ITYPE=3: error = U V**H - I
		if (*n) < 2 {
			return
		}
		golapack.Zlacpy(' ', n, n, u, ldu, work.CMatrix(*n, opts), n)
		if lower {
			golapack.Zunm2r('R', 'C', n, toPtr((*n)-1), toPtr((*n)-1), v.Off(1, 0), ldv, tau, work.CMatrixOff((*n), *n, opts), n, work.Off(pow(*n, 2)), &iinfo)
		} else {
			golapack.Zunm2l('R', 'C', n, toPtr((*n)-1), toPtr((*n)-1), v.Off(0, 1), ldv, tau, work.CMatrix(*n, opts), n, work.Off(pow(*n, 2)), &iinfo)
		}
		if iinfo != 0 {
			result.Set(0, ten/ulp)
			return
		}

		for j = 1; j <= (*n); j++ {
			work.Set(((*n)+1)*(j-1), work.Get(((*n)+1)*(j-1))-cone)
		}

		wnorm = golapack.Zlange('1', n, n, work.CMatrix(*n, opts), n, rwork)
	}

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(*n)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (math.Min(wnorm, float64(*n)*anorm)/anorm)/(float64(*n)*ulp))
		} else {
			result.Set(0, math.Min(wnorm/anorm, float64(*n))/(float64(*n)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U U**H - I
	if (*itype) == 1 {
		err = goblas.Zgemm(NoTrans, ConjTrans, *n, *n, *n, cone, u, u, czero, work.CMatrix(*n, opts))

		for j = 1; j <= (*n); j++ {
			work.Set(((*n)+1)*(j-1), work.Get(((*n)+1)*(j-1))-cone)
		}

		result.Set(1, math.Min(golapack.Zlange('1', n, n, work.CMatrix(*n, opts), n, rwork), float64(*n))/(float64(*n)*ulp))
	}
}
