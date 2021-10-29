package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zptt01 reconstructs a tridiagonal matrix A from its L*D*L'
// factorization and computes the residual
//    norm(L*D*L' - A) / ( n * norm(A) * EPS ),
// where EPS is the machine epsilon.
func zptt01(n int, d *mat.Vector, e *mat.CVector, df *mat.Vector, ef, work *mat.CVector) (resid float64) {
	var de complex128
	var anorm, eps, one, zero float64
	var i int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if n <= 0 {
		resid = zero
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Construct the difference L*D*L' - A.
	work.SetRe(0, df.Get(0)-d.Get(0))
	for i = 1; i <= n-1; i++ {
		de = df.GetCmplx(i-1) * ef.Get(i-1)
		work.Set(n+i-1, de-e.Get(i-1))
		work.Set(1+i-1, de*ef.GetConj(i-1)+df.GetCmplx(i)-d.GetCmplx(i))
	}

	//     Compute the 1-norms of the tridiagonal matrices A and WORK.
	if n == 1 {
		anorm = d.Get(0)
		resid = work.GetMag(0)
	} else {
		anorm = math.Max(d.Get(0)+e.GetMag(0), d.Get(n-1)+e.GetMag(n-1-1))
		resid = math.Max(work.GetMag(0)+work.GetMag(n), work.GetMag(n-1)+work.GetMag(2*n-1-1))
		for i = 2; i <= n-1; i++ {
			anorm = math.Max(anorm, d.Get(i-1)+e.GetMag(i-1)+e.GetMag(i-1-1))
			resid = math.Max(resid, work.GetMag(i-1)+work.GetMag(n+i-1-1)+work.GetMag(n+i-1))
		}
	}

	//     Compute norm(L*D*L' - A) / (n * norm(A) * EPS)
	if anorm <= zero {
		if resid != zero {
			resid = one / eps
		}
	} else {
		resid = ((resid / float64(n)) / anorm) / eps
	}

	return
}
