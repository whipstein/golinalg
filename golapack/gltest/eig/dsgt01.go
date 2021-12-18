package eig

import (
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dsgt01 checks a decomposition of the form
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
func dsgt01(itype int, uplo mat.MatUplo, n, m int, a, b, z *mat.Matrix, d, work, result *mat.Vector) {
	var anorm, one, ulp, zero float64
	var i int
	var err error

	zero = 0.0
	one = 1.0

	result.Set(0, zero)
	if n <= 0 {
		return
	}

	ulp = golapack.Dlamch(Epsilon)

	//     Compute product of 1-norms of A and Z.
	anorm = golapack.Dlansy('1', uplo, n, a, work) * golapack.Dlange('1', n, m, z, work)
	if anorm == zero {
		anorm = one
	}

	if itype == 1 {
		//        Norm of AZ - BZD
		if err = work.Matrix(n, opts).Symm(Left, uplo, n, m, one, a, z, zero); err != nil {
			panic(err)
		}
		for i = 1; i <= m; i++ {
			z.Off(0, i-1).Vector().Scal(n, d.Get(i-1), 1)
		}
		if err = work.Matrix(n, opts).Symm(Left, uplo, n, m, one, b, z, -one); err != nil {
			panic(err)
		}

		result.Set(0, (golapack.Dlange('1', n, m, work.Matrix(n, opts), work)/anorm)/(float64(n)*ulp))

	} else if itype == 2 {
		//        Norm of ABZ - ZD
		if err = work.Matrix(n, opts).Symm(Left, uplo, n, m, one, b, z, zero); err != nil {
			panic(err)
		}
		for i = 1; i <= m; i++ {
			z.Off(0, i-1).Vector().Scal(n, d.Get(i-1), 1)
		}
		if err = z.Symm(Left, uplo, n, m, one, a, work.Matrix(n, opts), -one); err != nil {
			panic(err)
		}

		result.Set(0, (golapack.Dlange('1', n, m, z, work)/anorm)/(float64(n)*ulp))

	} else if itype == 3 {
		//        Norm of BAZ - ZD
		if err = work.Matrix(n, opts).Symm(Left, uplo, n, m, one, a, z, zero); err != nil {
			panic(err)
		}
		for i = 1; i <= m; i++ {
			z.Off(0, i-1).Vector().Scal(n, d.Get(i-1), 1)
		}
		if err = z.Symm(Left, uplo, n, m, one, b, work.Matrix(n, opts), -one); err != nil {
			panic(err)
		}

		result.Set(0, (golapack.Dlange('1', n, m, z, work)/anorm)/(float64(n)*ulp))
	}
}
