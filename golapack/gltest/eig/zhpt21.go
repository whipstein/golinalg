package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zhpt21 generally checks a decomposition of the form
//
//         A = U S U**H
//
// where **H means conjugate transpose, A is hermitian, U is
// unitary, and S is diagonal (if KBAND=0) or (real) symmetric
// tridiagonal (if KBAND=1).  If ITYPE=1, then U is represented as
// a dense matrix, otherwise the U is expressed as a product of
// Householder transformations, whose vectors are stored in the
// array "V" and whose scaling constants are in "TAU"; we shall
// use the letter "V" to refer to the product of Householder
// transformations (which should be equal to U).
//
// Specifically, if ITYPE=1, then:
//
//         RESULT(1) = | A - U S U**H | / ( |A| n ulp ) and
//         RESULT(2) = | I - U U**H | / ( n ulp )
//
// If ITYPE=2, then:
//
//         RESULT(1) = | A - V S V**H | / ( |A| n ulp )
//
// If ITYPE=3, then:
//
//         RESULT(1) = | I - U V**H | / ( n ulp )
//
// Packed storage means that, for example, if UPLO='U', then the columns
// of the upper triangle of A are stored one after another, so that
// A(1,j+1) immediately follows A(j,j) in the array AP.  Similarly, if
// UPLO='L', then the columns of the lower triangle of A are stored one
// after another in AP, so that A(j+1,j+1) immediately follows A(n,j)
// in the array AP.  This means that A(i,j) is stored in:
//
//    AP( i + j*(j-1)/2 )                 if UPLO='U'
//
//    AP( i + (2*n-j)*(j-1)/2 )           if UPLO='L'
//
// The array VP bears the same relation to the matrix V that A does to
// AP.
//
// For ITYPE > 1, the transformation U is expressed as a product
// of Householder transformations:
//
//    If UPLO='U', then  V = H(n-1)...H(1),  where
//
//        H(j) = I  -  tau(j) v(j) v(j)**H
//
//    and the first j-1 elements of v(j) are stored in V(1:j-1,j+1),
//    (i.e., VP( j*(j+1)/2 + 1 : j*(j+1)/2 + j-1 ) ),
//    the j-th element is 1, and the last n-j elements are 0.
//
//    If UPLO='L', then  V = H(1)...H(n-1),  where
//
//        H(j) = I  -  tau(j) v(j) v(j)**H
//
//    and the first j elements of v(j) are 0, the (j+1)-st is 1, and the
//    (j+2)-nd through n-th elements are stored in V(j+2:n,j) (i.e.,
//    in VP( (2*n-j)*(j-1)/2 + j+2 : (2*n-j)*(j-1)/2 + n ) .)
func zhpt21(itype int, uplo mat.MatUplo, n, kband int, ap *mat.CVector, d, e *mat.Vector, u *mat.CMatrix, vp, tau, work *mat.CVector, rwork, result *mat.Vector) {
	var lower bool
	var cuplo mat.MatUplo
	var cone, czero, temp, vsave complex128
	var anorm, half, one, ten, ulp, unfl, wnorm, zero float64
	var j, jp, jp1, jr, lap int
	var err error

	zero = 0.0
	one = 1.0
	ten = 10.0
	half = 1.0 / 2.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Constants
	result.Set(0, zero)
	if itype == 1 {
		result.Set(1, zero)
	}
	if n <= 0 {
		return
	}

	lap = (n * (n + 1)) / 2

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
		anorm = math.Max(golapack.Zlanhp('1', cuplo, n, ap, rwork), unfl)
	}

	//     Compute error matrix:
	if itype == 1 {
		//        ITYPE=1: error = A - U S U**H
		golapack.Zlaset(Full, n, n, czero, czero, work.CMatrix(n, opts))
		goblas.Zcopy(lap, ap.Off(0, 1), work.Off(0, 1))

		for j = 1; j <= n; j++ {
			if err = goblas.Zhpr(cuplo, n, -d.Get(j-1), u.CVector(0, j-1, 1), work); err != nil {
				panic(err)
			}
		}

		if n > 1 && kband == 1 {
			for j = 1; j <= n-1; j++ {
				if err = goblas.Zhpr2(cuplo, n, -e.GetCmplx(j-1), u.CVector(0, j-1, 1), u.CVector(0, j-1-1, 1), work); err != nil {
					panic(err)
				}
			}
		}
		wnorm = golapack.Zlanhp('1', cuplo, n, work, rwork)

	} else if itype == 2 {
		//        ITYPE=2: error = V S V**H - A
		golapack.Zlaset(Full, n, n, czero, czero, work.CMatrix(n, opts))

		if lower {
			work.Set(lap-1, d.GetCmplx(n-1))
			for j = n - 1; j >= 1; j -= 1 {
				jp = ((2*n - j) * (j - 1)) / 2
				jp1 = jp + n - j
				if kband == 1 {
					work.Set(jp+j, (cone-tau.Get(j-1))*e.GetCmplx(j-1))
					for jr = j + 2; jr <= n; jr++ {
						work.Set(jp+jr-1, -tau.Get(j-1)*e.GetCmplx(j-1)*vp.Get(jp+jr-1))
					}
				}

				if tau.Get(j-1) != czero {
					vsave = vp.Get(jp + j + 1 - 1)
					vp.Set(jp+j, cone)
					err = goblas.Zhpmv(Lower, n-j, cone, work.Off(jp1+j), vp.Off(jp+j, 1), czero, work.Off(lap, 1))
					temp = complex(-half, 0) * tau.Get(j-1) * goblas.Zdotc(n-j, work.Off(lap, 1), vp.Off(jp+j, 1))
					goblas.Zaxpy(n-j, temp, vp.Off(jp+j, 1), work.Off(lap, 1))
					err = goblas.Zhpr2(Lower, n-j, -tau.Get(j-1), vp.Off(jp+j, 1), work.Off(lap, 1), work.Off(jp1+j))

					vp.Set(jp+j, vsave)
				}
				work.Set(jp+j-1, d.GetCmplx(j-1))
			}
		} else {
			work.Set(0, d.GetCmplx(0))
			for j = 1; j <= n-1; j++ {
				jp = (j * (j - 1)) / 2
				jp1 = jp + j
				if kband == 1 {
					work.Set(jp1+j-1, (cone-tau.Get(j-1))*e.GetCmplx(j-1))
					for jr = 1; jr <= j-1; jr++ {
						work.Set(jp1+jr-1, -tau.Get(j-1)*e.GetCmplx(j-1)*vp.Get(jp1+jr-1))
					}
				}

				if tau.Get(j-1) != czero {
					vsave = vp.Get(jp1 + j - 1)
					vp.Set(jp1+j-1, cone)
					err = goblas.Zhpmv(Upper, j, cone, work, vp.Off(jp1, 1), czero, work.Off(lap, 1))
					temp = complex(-half, 0) * tau.Get(j-1) * goblas.Zdotc(j, work.Off(lap, 1), vp.Off(jp1, 1))
					goblas.Zaxpy(j, temp, vp.Off(jp1, 1), work.Off(lap, 1))
					err = goblas.Zhpr2(Upper, j, -tau.Get(j-1), vp.Off(jp1, 1), work.Off(lap, 1), work)
					vp.Set(jp1+j-1, vsave)
				}
				work.Set(jp1+j, d.GetCmplx(j))
			}
		}

		for j = 1; j <= lap; j++ {
			work.Set(j-1, work.Get(j-1)-ap.Get(j-1))
		}
		wnorm = golapack.Zlanhp('1', cuplo, n, work, rwork)

	} else if itype == 3 {
		//        ITYPE=3: error = U V**H - I
		if n < 2 {
			return
		}
		golapack.Zlacpy(Full, n, n, u, work.CMatrix(n, opts))
		if err = golapack.Zupmtr(Right, cuplo, ConjTrans, n, n, vp, tau, work.CMatrix(n, opts), work.Off(pow(n, 2))); err != nil {
			result.Set(0, ten/ulp)
			return
		}

		for j = 1; j <= n; j++ {
			work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-cone)
		}

		wnorm = golapack.Zlange('1', n, n, work.CMatrix(n, opts), rwork)
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
	//     Compute  U U**H - I
	if itype == 1 {
		if err = goblas.Zgemm(NoTrans, ConjTrans, n, n, n, cone, u, u, czero, work.CMatrix(n, opts)); err != nil {
			panic(err)
		}

		for j = 1; j <= n; j++ {
			work.Set((n+1)*(j-1), work.Get((n+1)*(j-1))-cone)
		}

		result.Set(1, math.Min(golapack.Zlange('1', n, n, work.CMatrix(n, opts), rwork), float64(n))/(float64(n)*ulp))
	}
}
