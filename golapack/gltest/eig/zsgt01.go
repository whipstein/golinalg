package eig

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zsgt01 checks a decomposition of the form
//
//    A Z   =  B Z D or
//    A B Z =  Z D or
//    B A Z =  Z D
//
// where A is a Hermitian matrix, B is Hermitian positive definite,
// Z is unitary, and D is diagonal.
//
// One of the following test ratios is computed:
//
// ITYPE = 1:  RESULT(1) = | A Z - B Z D | / ( |A| |Z| n ulp )
//
// ITYPE = 2:  RESULT(1) = | A B Z - Z D | / ( |A| |Z| n ulp )
//
// ITYPE = 3:  RESULT(1) = | B A Z - Z D | / ( |A| |Z| n ulp )
func Zsgt01(itype *int, uplo byte, n, m *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, z *mat.CMatrix, ldz *int, d *mat.Vector, work *mat.CVector, rwork, result *mat.Vector) {
	var cone, czero complex128
	var anorm, one, ulp, zero float64
	var i int
	var err error
	_ = err

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	result.Set(0, zero)
	if (*n) <= 0 {
		return
	}

	ulp = golapack.Dlamch(Epsilon)

	//     Compute product of 1-norms of A and Z.
	anorm = golapack.Zlanhe('1', uplo, n, a, lda, rwork) * golapack.Zlange('1', n, m, z, ldz, rwork)
	if anorm == zero {
		anorm = one
	}

	if (*itype) == 1 {
		//        Norm of AZ - BZD
		err = goblas.Zhemm(Left, mat.UploByte(uplo), *n, *m, cone, a, z, czero, work.CMatrix(*n, opts))
		for i = 1; i <= (*m); i++ {
			goblas.Zdscal(*n, d.Get(i-1), z.CVector(0, i-1, 1))
		}
		err = goblas.Zhemm(Left, mat.UploByte(uplo), *n, *m, cone, b, z, -cone, work.CMatrix(*n, opts))

		result.Set(0, (golapack.Zlange('1', n, m, work.CMatrix(*n, opts), n, rwork)/anorm)/(float64(*n)*ulp))

	} else if (*itype) == 2 {
		//        Norm of ABZ - ZD
		err = goblas.Zhemm(Left, mat.UploByte(uplo), *n, *m, cone, b, z, czero, work.CMatrix(*n, opts))
		for i = 1; i <= (*m); i++ {
			goblas.Zdscal(*n, d.Get(i-1), z.CVector(0, i-1, 1))
		}
		err = goblas.Zhemm(Left, mat.UploByte(uplo), *n, *m, cone, a, work.CMatrix(*n, opts), -cone, z)

		result.Set(0, (golapack.Zlange('1', n, m, z, ldz, rwork)/anorm)/(float64(*n)*ulp))

	} else if (*itype) == 3 {
		//        Norm of BAZ - ZD
		err = goblas.Zhemm(Left, mat.UploByte(uplo), *n, *m, cone, a, z, czero, work.CMatrix(*n, opts))
		for i = 1; i <= (*m); i++ {
			goblas.Zdscal(*n, d.Get(i-1), z.CVector(0, i-1, 1))
		}
		err = goblas.Zhemm(Left, mat.UploByte(uplo), *n, *m, cone, b, work.CMatrix(*n, opts), -cone, z)

		result.Set(0, (golapack.Zlange('1', n, m, z, ldz, rwork)/anorm)/(float64(*n)*ulp))
	}
}
