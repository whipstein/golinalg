package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Dgbt01 reconstructs a band matrix  A  from its L*U factorization and
// computes the residual:
//    norm(L*U - A) / ( N * norm(A) * EPS ),
// where EPS is the machine epsilon.
//
// The expression L*U - A is computed one column at a time, so A and
// AFAC are not modified.
func Dgbt01(m, n, kl, ku *int, a *mat.Matrix, lda *int, afac *mat.Matrix, ldafac *int, ipiv *[]int, work *mat.Vector, resid *float64) {
	var anorm, eps, one, t, zero float64
	var i, i1, i2, il, ip, iw, j, jl, ju, jua, kd, lenj int

	zero = 0.0
	one = 1.0

	//     Quick exit if M = 0 or N = 0.
	(*resid) = zero
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	//     Determine EPS and the norm of A.
	eps = golapack.Dlamch(Epsilon)
	kd = (*ku) + 1
	anorm = zero
	for j = 1; j <= (*n); j++ {
		i1 = maxint(kd+1-j, 1)
		i2 = minint(kd+(*m)-j, (*kl)+kd)
		if i2 >= i1 {
			anorm = maxf64(anorm, goblas.Dasum(toPtr(i2-i1+1), a.Vector(i1-1, j-1), toPtr(1)))
		}
	}

	//     Compute one column at a time of L*U - A.
	kd = (*kl) + (*ku) + 1
	for j = 1; j <= (*n); j++ {
		//        Copy the J-th column of U to WORK.
		ju = minint((*kl)+(*ku), j-1)
		jl = minint(*kl, (*m)-j)
		lenj = minint(*m, j) - j + ju + 1
		if lenj > 0 {
			goblas.Dcopy(&lenj, afac.Vector(kd-ju-1, j-1), toPtr(1), work, toPtr(1))
			for i = lenj + 1; i <= ju+jl+1; i++ {
				work.Set(i-1, zero)
			}

			//           Multiply by the unit lower triangular matrix L.  Note that L
			//           is stored as a product of transformations and permutations.
			for i = minint((*m)-1, j); i >= j-ju; i-- {
				il = minint(*kl, (*m)-i)
				if il > 0 {
					iw = i - j + ju + 1
					t = work.Get(iw - 1)
					goblas.Daxpy(&il, &t, afac.Vector(kd+1-1, i-1), toPtr(1), work.Off(iw+1-1), toPtr(1))
					ip = (*ipiv)[i-1]
					if i != ip {
						ip = ip - j + ju + 1
						work.Set(iw-1, work.Get(ip-1))
						work.Set(ip-1, t)
					}
				}
			}

			//           Subtract the corresponding column of A.
			jua = minint(ju, *ku)
			if jua+jl+1 > 0 {
				goblas.Daxpy(toPtr(jua+jl+1), toPtrf64(-one), a.Vector((*ku)+1-jua-1, j-1), toPtr(1), work.Off(ju+1-jua-1), toPtr(1))
			}

			//           Compute the 1-norm of the column.
			(*resid) = maxf64(*resid, goblas.Dasum(toPtr(ju+jl+1), work, toPtr(1)))
		}
	}

	//     Compute norm( L*U - A ) / ( N * norm(A) * EPS )
	if anorm <= zero {
		if (*resid) != zero {
			(*resid) = one / eps
		}
	} else {
		(*resid) = (((*resid) / float64(*n)) / anorm) / eps
	}
}
