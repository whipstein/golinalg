package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlaed5 subroutine computes the I-th eigenvalue of a symmetric rank-one
// modification of a 2-by-2 diagonal matrix
//
//            diag( D )  +  RHO * Z * transpose(Z) .
//
// The diagonal elements in the array D are assumed to satisfy
//
//            D(i) < D(j)  for  i < j .
//
// We also assume RHO > 0 and that the Euclidean norm of the vector
// Z is one.
func Dlaed5(i *int, d, z, delta *mat.Vector, rho, dlam *float64) {
	var b, c, del, four, one, tau, temp, two, w, zero float64

	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0

	del = d.Get(1) - d.Get(0)
	if (*i) == 1 {
		w = one + two*(*rho)*(z.Get(1)*z.Get(1)-z.Get(0)*z.Get(0))/del
		if w > zero {
			b = del + (*rho)*(z.Get(0)*z.Get(0)+z.Get(1)*z.Get(1))
			c = (*rho) * z.Get(0) * z.Get(0) * del

			//           B > ZERO, always
			tau = two * c / (b + math.Sqrt(math.Abs(b*b-four*c)))
			(*dlam) = d.Get(0) + tau
			delta.Set(0, -z.Get(0)/tau)
			delta.Set(1, z.Get(1)/(del-tau))
		} else {
			b = -del + (*rho)*(z.Get(0)*z.Get(0)+z.Get(1)*z.Get(1))
			c = (*rho) * z.Get(1) * z.Get(1) * del
			if b > zero {
				tau = -two * c / (b + math.Sqrt(b*b+four*c))
			} else {
				tau = (b - math.Sqrt(b*b+four*c)) / two
			}
			(*dlam) = d.Get(1) + tau
			delta.Set(0, -z.Get(0)/(del+tau))
			delta.Set(1, -z.Get(1)/tau)
		}
		temp = math.Sqrt(delta.Get(0)*delta.Get(0) + delta.Get(1)*delta.Get(1))
		delta.Set(0, delta.Get(0)/temp)
		delta.Set(1, delta.Get(1)/temp)
	} else {
		//     Now I=2
		b = -del + (*rho)*(z.Get(0)*z.Get(0)+z.Get(1)*z.Get(1))
		c = (*rho) * z.Get(1) * z.Get(1) * del
		if b > zero {
			tau = (b + math.Sqrt(b*b+four*c)) / two
		} else {
			tau = two * c / (-b + math.Sqrt(b*b+four*c))
		}
		(*dlam) = d.Get(1) + tau
		delta.Set(0, -z.Get(0)/(del+tau))
		delta.Set(1, -z.Get(1)/tau)
		temp = math.Sqrt(delta.Get(0)*delta.Get(0) + delta.Get(1)*delta.Get(1))
		delta.Set(0, delta.Get(0)/temp)
		delta.Set(1, delta.Get(1)/temp)
	}
}
