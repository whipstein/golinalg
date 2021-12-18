package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// dstt22 checks a set of M eigenvalues and eigenvectors,
//
//     A U = U S
//
// where A is symmetric tridiagonal, the columns of U are orthogonal,
// and S is diagonal (if KBAND=0) or symmetric tridiagonal (if KBAND=1).
// Two tests are performed:
//
//    RESULT(1) = | U' A U - S | / ( |A| m ulp )
//
//    RESULT(2) = | I - U'U | / ( m ulp )
func dstt22(n, m, kband int, ad, ae, sd, se *mat.Vector, u, work *mat.Matrix, result *mat.Vector) {
	var anorm, aukj, one, ulp, unfl, wnorm, zero float64
	var i, j, k int
	var err error

	zero = 0.0
	one = 1.0

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
		anorm = math.Abs(ad.Get(0)) + math.Abs(ae.Get(0))
		for j = 2; j <= n-1; j++ {
			anorm = math.Max(anorm, math.Abs(ad.Get(j-1))+math.Abs(ae.Get(j-1))+math.Abs(ae.Get(j-1-1)))
		}
		anorm = math.Max(anorm, math.Abs(ad.Get(n-1))+math.Abs(ae.Get(n-1-1)))
	} else {
		anorm = math.Abs(ad.Get(0))
	}
	anorm = math.Max(anorm, unfl)

	//     Norm of U'AU - S
	for i = 1; i <= m; i++ {
		for j = 1; j <= m; j++ {
			work.Set(i-1, j-1, zero)
			for k = 1; k <= n; k++ {
				aukj = ad.Get(k-1) * u.Get(k-1, j-1)
				if k != n {
					aukj = aukj + ae.Get(k-1)*u.Get(k, j-1)
				}
				if k != 1 {
					aukj = aukj + ae.Get(k-1-1)*u.Get(k-1-1, j-1)
				}
				work.Set(i-1, j-1, work.Get(i-1, j-1)+u.Get(k-1, i-1)*aukj)
			}
		}
		work.Set(i-1, i-1, work.Get(i-1, i-1)-sd.Get(i-1))
		if kband == 1 {
			if i != 1 {
				work.Set(i-1, i-1-1, work.Get(i-1, i-1-1)-se.Get(i-1-1))
			}
			if i != n {
				work.Set(i-1, i, work.Get(i-1, i)-se.Get(i-1))
			}
		}
	}

	wnorm = golapack.Dlansy('1', Lower, m, work, work.Off(0, m).Vector())

	if anorm > wnorm {
		result.Set(0, (wnorm/anorm)/(float64(m)*ulp))
	} else {
		if anorm < one {
			result.Set(0, (math.Min(wnorm, float64(m)*anorm)/anorm)/(float64(m)*ulp))
		} else {
			result.Set(0, math.Min(wnorm/anorm, float64(m))/(float64(m)*ulp))
		}
	}

	//     Do Test 2
	//
	//     Compute  U'U - I
	if err = work.Gemm(Trans, NoTrans, m, m, n, one, u, u, zero); err != nil {
		panic(err)
	}

	for j = 1; j <= m; j++ {
		work.Set(j-1, j-1, work.Get(j-1, j-1)-one)
	}

	result.Set(1, math.Min(float64(m), golapack.Dlange('1', m, m, work, work.Off(0, m).Vector()))/(float64(m)*ulp))
}
