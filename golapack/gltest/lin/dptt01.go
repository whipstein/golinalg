package lin

import (
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dptt01 reconstructs a tridiagonal matrix A from its L*D*L'
// factorization and computes the residual
//    norm(L*D*L' - A) / ( n * norm(A) * EPS ),
// where EPS is the machine epsilon.
func Dptt01(n *int, d, e, df, ef, work *mat.Vector, resid *float64) {
	var anorm, de, eps, one, zero float64
	var i int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Construct the difference L*D*L' - A.
	work.Set(0, df.Get(0)-d.Get(0))
	for i = 1; i <= (*n)-1; i++ {
		de = df.Get(i-1) * ef.Get(i-1)
		work.Set((*n)+i-1, de-e.Get(i-1))
		work.Set(1+i-1, de*ef.Get(i-1)+df.Get(i+1-1)-d.Get(i+1-1))
	}

	//     Compute the 1-norms of the tridiagonal matrices A and WORK.
	if (*n) == 1 {
		anorm = d.Get(0)
		(*resid) = math.Abs(work.Get(0))
	} else {
		anorm = maxf64(d.Get(0)+math.Abs(e.Get(0)), d.Get((*n)-1)+math.Abs(e.Get((*n)-1-1)))
		(*resid) = maxf64(math.Abs(work.Get(0))+math.Abs(work.Get((*n)+1-1)), math.Abs(work.Get((*n)-1))+math.Abs(work.Get(2*(*n)-1-1)))
		for i = 2; i <= (*n)-1; i++ {
			anorm = maxf64(anorm, d.Get(i-1)+math.Abs(e.Get(i-1))+math.Abs(e.Get(i-1-1)))
			(*resid) = maxf64(*resid, math.Abs(work.Get(i-1))+math.Abs(work.Get((*n)+i-1-1))+math.Abs(work.Get((*n)+i-1)))
		}
	}

	//     Compute norm(L*D*L' - A) / (n * norm(A) * EPS)
	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}
}
