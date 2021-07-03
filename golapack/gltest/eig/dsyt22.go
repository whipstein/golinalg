package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dsyt22 generally checks a decomposition of the form
//
//              A U = U S
//
//      where A is symmetric, the columns of U are orthonormal, and S
//      is diagonal (if KBAND=0) or symmetric tridiagonal (if
//      KBAND=1).  If ITYPE=1, then U is represented as a dense matrix,
//      otherwise the U is expressed as a product of Householder
//      transformations, whose vectors are stored in the array "V" and
//      whose scaling constants are in "TAU"; we shall use the letter
//      "V" to refer to the product of Householder transformations
//      (which should be equal to U).
//
//      Specifically, if ITYPE=1, then:
//
//              RESULT(1) = | U**T A U - S | / ( |A| m ulp ) and
//              RESULT(2) = | I - U**T U | / ( m ulp )
func Dsyt22(itype *int, uplo byte, n, m, kband *int, a *mat.Matrix, lda *int, d, e *mat.Vector, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, tau, work, result *mat.Vector) {
	var anorm, one, ulp, unfl, wnorm, zero float64
	var j, jj, jj1, jj2, nn, nnp1 int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	result.Set(0, zero)
	result.Set(1, zero)
	if (*n) <= 0 || (*m) <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Precision)

	//     Do Test 1
	//
	//     Norm of A:
	anorm = maxf64(golapack.Dlansy('1', uplo, n, a, lda, work), unfl)

	//     Compute error matrix:
	//
	//     ITYPE=1: error = U**T A U - S
	err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, a, *lda, u, *ldu, zero, work.Matrix(*n, opts), *n)
	nn = (*n) * (*n)
	nnp1 = nn + 1
	err = goblas.Dgemm(Trans, NoTrans, *m, *m, *n, one, u, *ldu, work.Matrix(*n, opts), *n, zero, work.MatrixOff(nnp1-1, *n, opts), *n)
	for j = 1; j <= (*m); j++ {
		jj = nn + (j-1)*(*n) + j
		work.Set(jj-1, work.Get(jj-1)-d.Get(j-1))
	}
	if (*kband) == 1 && (*n) > 1 {
		for j = 2; j <= (*m); j++ {
			jj1 = nn + (j-1)*(*n) + j - 1
			jj2 = nn + (j-2)*(*n) + j
			work.Set(jj1-1, work.Get(jj1-1)-e.Get(j-1-1))
			work.Set(jj2-1, work.Get(jj2-1)-e.Get(j-1-1))
		}
	}
	wnorm = golapack.Dlansy('1', uplo, m, work.MatrixOff(nnp1-1, *n, opts), n, work)

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(*m)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (minf64(wnorm, float64(*m)*anorm)/anorm)/(float64(*m)*ulp))
		} else {
			result.Set(0, minf64(wnorm/anorm, float64(*m))/(float64(*m)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U**T U - I
	if (*itype) == 1 {
		Dort01('C', n, m, u, ldu, work, toPtr(2*(*n)*(*n)), result.GetPtr(1))
	}
}
