package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zgtt01 reconstructs a tridiagonal matrix A from its LU factorization
// and computes the residual
//    norm(L*U - A) / ( norm(A) * EPS ),
// where EPS is the machine epsilon.
func Zgtt01(n *int, dl, d, du, dlf, df, duf, du2 *mat.CVector, ipiv *[]int, work *mat.CMatrix, ldwork *int, rwork *mat.Vector, resid *float64) {
	var li complex128
	var anorm, eps, one, zero float64
	var i, ip, j, lastj int

	one = 1.0
	zero = 0.0

	//     Quick return if possible
	if (*n) <= 0 {
		(*resid) = zero
		return
	}

	eps = golapack.Dlamch(Epsilon)

	//     Copy the matrix U to WORK.
	for j = 1; j <= (*n); j++ {
		for i = 1; i <= (*n); i++ {
			work.SetRe(i-1, j-1, zero)
		}
	}
	for i = 1; i <= (*n); i++ {
		if i == 1 {
			work.Set(i-1, i-1, df.Get(i-1))
			if (*n) >= 2 {
				work.Set(i-1, i+1-1, duf.Get(i-1))
			}
			if (*n) >= 3 {
				work.Set(i-1, i+2-1, du2.Get(i-1))
			}
		} else if i == (*n) {
			work.Set(i-1, i-1, df.Get(i-1))
		} else {
			work.Set(i-1, i-1, df.Get(i-1))
			work.Set(i-1, i+1-1, duf.Get(i-1))
			if i < (*n)-1 {
				work.Set(i-1, i+2-1, du2.Get(i-1))
			}
		}
	}

	//     Multiply on the left by L.
	lastj = (*n)
	for i = (*n) - 1; i >= 1; i-- {
		li = dlf.Get(i - 1)
		goblas.Zaxpy(lastj-i+1, li, work.CVector(i-1, i-1), *ldwork, work.CVector(i+1-1, i-1), *ldwork)
		ip = (*ipiv)[i-1]
		if ip == i {
			lastj = minint(i+2, *n)
		} else {
			goblas.Zswap(lastj-i+1, work.CVector(i-1, i-1), *ldwork, work.CVector(i+1-1, i-1), *ldwork)
		}
	}

	//     Subtract the matrix A.
	work.Set(0, 0, work.Get(0, 0)-d.Get(0))
	if (*n) > 1 {
		work.Set(0, 1, work.Get(0, 1)-du.Get(0))
		work.Set((*n)-1, (*n)-1-1, work.Get((*n)-1, (*n)-1-1)-dl.Get((*n)-1-1))
		work.Set((*n)-1, (*n)-1, work.Get((*n)-1, (*n)-1)-d.Get((*n)-1))
		for i = 2; i <= (*n)-1; i++ {
			work.Set(i-1, i-1-1, work.Get(i-1, i-1-1)-dl.Get(i-1-1))
			work.Set(i-1, i-1, work.Get(i-1, i-1)-d.Get(i-1))
			work.Set(i-1, i+1-1, work.Get(i-1, i+1-1)-du.Get(i-1))
		}
	}

	//     Compute the 1-norm of the tridiagonal matrix A.
	anorm = golapack.Zlangt('1', n, dl, d, du)

	//     Compute the 1-norm of WORK, which is only guaranteed to be
	//     upper Hessenberg.
	(*resid) = golapack.Zlanhs('1', n, work, ldwork, rwork)

	//     Compute norm(L*U - A) / (norm(A) * EPS)
	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = ((*resid) / anorm) / eps
	}
}
