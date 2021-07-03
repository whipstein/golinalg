package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dsgt01 checks a decomposition of the form
//
//    A Z   =  B Z D or
//    A B Z =  Z D or
//    B A Z =  Z D
//
// where A is a symmetric matrix, B is
// symmetric positive definite, Z is orthogonal, and D is diagonal.
//
// One of the following test ratios is computed:
//
// ITYPE = 1:  RESULT(1) = | A Z - B Z D | / ( |A| |Z| n ulp )
//
// ITYPE = 2:  RESULT(1) = | A B Z - Z D | / ( |A| |Z| n ulp )
//
// ITYPE = 3:  RESULT(1) = | B A Z - Z D | / ( |A| |Z| n ulp )
func Dsgt01(itype *int, uplo byte, n, m *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, z *mat.Matrix, ldz *int, d, work, result *mat.Vector) {
	var anorm, one, ulp, zero float64
	var i int
	var err error
	_ = err

	zero = 0.0
	one = 1.0

	result.Set(0, zero)
	if (*n) <= 0 {
		return
	}

	ulp = golapack.Dlamch(Epsilon)

	//     Compute product of 1-norms of A and Z.
	anorm = golapack.Dlansy('1', uplo, n, a, lda, work) * golapack.Dlange('1', n, m, z, ldz, work)
	if anorm == zero {
		anorm = one
	}

	if (*itype) == 1 {
		//        Norm of AZ - BZD
		err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, a, *lda, z, *ldz, zero, work.Matrix(*n, opts), *n)
		for i = 1; i <= (*m); i++ {
			goblas.Dscal(*n, d.Get(i-1), z.Vector(0, i-1), 1)
		}
		err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, b, *ldb, z, *ldz, -one, work.Matrix(*n, opts), *n)

		result.Set(0, (golapack.Dlange('1', n, m, work.Matrix(*n, opts), n, work)/anorm)/(float64(*n)*ulp))

	} else if (*itype) == 2 {
		//        Norm of ABZ - ZD
		err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, b, *ldb, z, *ldz, zero, work.Matrix(*n, opts), *n)
		for i = 1; i <= (*m); i++ {
			goblas.Dscal(*n, d.Get(i-1), z.Vector(0, i-1), 1)
		}
		err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, a, *lda, work.Matrix(*n, opts), *n, -one, z, *ldz)

		result.Set(0, (golapack.Dlange('1', n, m, z, ldz, work)/anorm)/(float64(*n)*ulp))

	} else if (*itype) == 3 {
		//        Norm of BAZ - ZD
		err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, a, *lda, z, *ldz, zero, work.Matrix(*n, opts), *n)
		for i = 1; i <= (*m); i++ {
			goblas.Dscal(*n, d.Get(i-1), z.Vector(0, i-1), 1)
		}
		err = goblas.Dsymm(Left, mat.UploByte(uplo), *n, *m, one, b, *ldb, work.Matrix(*n, opts), *n, -one, z, *ldz)

		result.Set(0, (golapack.Dlange('1', n, m, z, ldz, work)/anorm)/(float64(*n)*ulp))
	}
}
