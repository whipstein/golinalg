package eig

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zsgt01 checks a decomposition of the form
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
func zsgt01(itype int, uplo mat.MatUplo, n, m int, a, b, z *mat.CMatrix, d *mat.Vector, work *mat.CVector, rwork, result *mat.Vector) {
	var cone, czero complex128
	var anorm, one, ulp, zero float64
	var i int
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	result.Set(0, zero)
	if n <= 0 {
		return
	}

	ulp = golapack.Dlamch(Epsilon)

	//     Compute product of 1-norms of A and Z.
	anorm = golapack.Zlanhe('1', uplo, n, a, rwork) * golapack.Zlange('1', n, m, z, rwork)
	if anorm == zero {
		anorm = one
	}

	if itype == 1 {
		//        Norm of AZ - BZD
		if err = work.CMatrix(n, opts).Hemm(Left, uplo, n, m, cone, a, z, czero); err != nil {
			panic(err)
		}
		for i = 1; i <= m; i++ {
			z.Off(0, i-1).CVector().Dscal(n, d.Get(i-1), 1)
		}
		if err = work.CMatrix(n, opts).Hemm(Left, uplo, n, m, cone, b, z, -cone); err != nil {
			panic(err)
		}

		result.Set(0, (golapack.Zlange('1', n, m, work.CMatrix(n, opts), rwork)/anorm)/(float64(n)*ulp))

	} else if itype == 2 {
		//        Norm of ABZ - ZD
		if err = work.CMatrix(n, opts).Hemm(Left, uplo, n, m, cone, b, z, czero); err != nil {
			panic(err)
		}
		for i = 1; i <= m; i++ {
			z.Off(0, i-1).CVector().Dscal(n, d.Get(i-1), 1)
		}
		if err = z.Hemm(Left, uplo, n, m, cone, a, work.CMatrix(n, opts), -cone); err != nil {
			panic(err)
		}

		result.Set(0, (golapack.Zlange('1', n, m, z, rwork)/anorm)/(float64(n)*ulp))

	} else if itype == 3 {
		//        Norm of BAZ - ZD
		if err = work.CMatrix(n, opts).Hemm(Left, uplo, n, m, cone, a, z, czero); err != nil {
			panic(err)
		}
		for i = 1; i <= m; i++ {
			z.Off(0, i-1).CVector().Dscal(n, d.Get(i-1), 1)
		}
		if err = z.Hemm(Left, uplo, n, m, cone, b, work.CMatrix(n, opts), -cone); err != nil {
			panic(err)
		}

		result.Set(0, (golapack.Zlange('1', n, m, z, rwork)/anorm)/(float64(n)*ulp))
	}
}
