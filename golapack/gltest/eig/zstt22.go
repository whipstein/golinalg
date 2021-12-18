package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zstt22 checks a set of M eigenvalues and eigenvectors,
//
//     A U = U S
//
// where A is Hermitian tridiagonal, the columns of U are unitary,
// and S is diagonal (if KBAND=0) or Hermitian tridiagonal (if KBAND=1).
// Two tests are performed:
//
//    RESULT(1) = | U* A U - S | / ( |A| m ulp )
//
//    RESULT(2) = | I - U*U | / ( m ulp )
func zstt22(n, m, kband int, ad, ae, sd, se *mat.Vector, u, work *mat.CMatrix, rwork, result *mat.Vector) {
	var aukj, cone, czero complex128
	var anorm, one, ulp, unfl, wnorm, zero float64
	var i, j, k int
	var err error

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	result.Set(0, zero)
	result.Set(1, zero)
	if n <= 0 || m <= 0 {
		return
	}

	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon)

	//     Do Test 1
	//
	//     Compute the 1-norm of A.
	if n > 1 {
		anorm = ad.GetMag(0) + ae.GetMag(0)
		for j = 2; j <= n-1; j++ {
			anorm = math.Max(anorm, ad.GetMag(j-1)+ae.GetMag(j-1)+ae.GetMag(j-1-1))
		}
		anorm = math.Max(anorm, ad.GetMag(n-1)+ae.GetMag(n-1-1))
	} else {
		anorm = ad.GetMag(0)
	}
	anorm = math.Max(anorm, unfl)

	//     Norm of U*AU - S
	for i = 1; i <= m; i++ {
		for j = 1; j <= m; j++ {
			work.Set(i-1, j-1, czero)
			for k = 1; k <= n; k++ {
				aukj = ad.GetCmplx(k-1) * u.Get(k-1, j-1)
				if k != n {
					aukj = aukj + ae.GetCmplx(k-1)*u.Get(k, j-1)
				}
				if k != 1 {
					aukj = aukj + ae.GetCmplx(k-1-1)*u.Get(k-1-1, j-1)
				}
				work.Set(i-1, j-1, work.Get(i-1, j-1)+u.Get(k-1, i-1)*aukj)
			}
		}
		work.Set(i-1, i-1, work.Get(i-1, i-1)-sd.GetCmplx(i-1))
		if kband == 1 {
			if i != 1 {
				work.Set(i-1, i-1-1, work.Get(i-1, i-1-1)-se.GetCmplx(i-1-1))
			}
			if i != n {
				work.Set(i-1, i, work.Get(i-1, i)-se.GetCmplx(i-1))
			}
		}
	}

	wnorm = golapack.Zlansy('1', Lower, m, work, rwork)

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
	//     Compute  U*U - I
	if err = work.Gemm(Trans, NoTrans, m, m, n, cone, u, u, czero); err != nil {
		panic(err)
	}

	for j = 1; j <= m; j++ {
		work.Set(j-1, j-1, work.Get(j-1, j-1)-complex(one, 0))
	}

	result.Set(1, math.Min(float64(n), golapack.Zlange('1', m, m, work, rwork))/(float64(n)*ulp))
}
