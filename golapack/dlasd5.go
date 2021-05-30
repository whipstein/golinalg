package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlasd5 subroutine computes the square root of the I-th eigenvalue
// of a positive symmetric rank-one modification of a 2-by-2 diagonal
// matrix
//
//            diag( D ) * diag( D ) +  RHO * Z * transpose(Z) .
//
// The diagonal entries in the array D are assumed to satisfy
//
//            0 <= D(i) < D(j)  for  i < j .
//
// We also assume RHO > 0 and that the Euclidean norm of the vector
// Z is one.
func Dlasd5(i *int, d, z, delta *mat.Vector, rho, dsigma *float64, work *mat.Vector) {
	var b, c, del, delsq, four, one, tau, three, two, w, zero float64

	zero = 0.0
	one = 1.0
	two = 2.0
	three = 3.0
	four = 4.0

	del = d.Get(1) - d.Get(0)
	delsq = del * (d.Get(1) + d.Get(0))
	if (*i) == 1 {
		w = one + four*(*rho)*(z.Get(1)*z.Get(1)/(d.Get(0)+three*d.Get(1))-z.Get(0)*z.Get(0)/(three*d.Get(0)+d.Get(1)))/del
		if w > zero {
			b = delsq + (*rho)*(z.Get(0)*z.Get(0)+z.Get(1)*z.Get(1))
			c = (*rho) * z.Get(0) * z.Get(0) * delsq

			//           B > ZERO, always
			//
			//           The following TAU is DSIGMA * DSIGMA - D( 1 ) * D( 1 )
			tau = two * c / (b + math.Sqrt(math.Abs(b*b-four*c)))

			//           The following TAU is DSIGMA - D( 1 )
			tau = tau / (d.Get(0) + math.Sqrt(d.Get(0)*d.Get(0)+tau))
			(*dsigma) = d.Get(0) + tau
			delta.Set(0, -tau)
			delta.Set(1, del-tau)
			work.Set(0, two*d.Get(0)+tau)
			work.Set(1, (d.Get(0)+tau)+d.Get(1))
			//           DELTA( 1 ) = -Z( 1 ) / TAU
			//           DELTA( 2 ) = Z( 2 ) / ( DEL-TAU )
		} else {
			b = -delsq + (*rho)*(z.Get(0)*z.Get(0)+z.Get(1)*z.Get(1))
			c = (*rho) * z.Get(1) * z.Get(1) * delsq

			//           The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 )
			if b > zero {
				tau = -two * c / (b + math.Sqrt(b*b+four*c))
			} else {
				tau = (b - math.Sqrt(b*b+four*c)) / two
			}

			//           The following TAU is DSIGMA - D( 2 )
			tau = tau / (d.Get(1) + math.Sqrt(math.Abs(d.Get(1)*d.Get(1)+tau)))
			(*dsigma) = d.Get(1) + tau
			delta.Set(0, -(del + tau))
			delta.Set(1, -tau)
			work.Set(0, d.Get(0)+tau+d.Get(1))
			work.Set(1, two*d.Get(1)+tau)
			//           DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
			//           DELTA( 2 ) = -Z( 2 ) / TAU
		}
		//        TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
		//        DELTA( 1 ) = DELTA( 1 ) / TEMP
		//        DELTA( 2 ) = DELTA( 2 ) / TEMP
	} else {
		//        Now I=2
		b = -delsq + (*rho)*(z.Get(0)*z.Get(0)+z.Get(1)*z.Get(1))
		c = (*rho) * z.Get(1) * z.Get(1) * delsq

		//        The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 )
		if b > zero {
			tau = (b + math.Sqrt(b*b+four*c)) / two
		} else {
			tau = two * c / (-b + math.Sqrt(b*b+four*c))
		}

		//        The following TAU is DSIGMA - D( 2 )
		tau = tau / (d.Get(1) + math.Sqrt(d.Get(1)*d.Get(1)+tau))
		(*dsigma) = d.Get(1) + tau
		delta.Set(0, -(del + tau))
		delta.Set(1, -tau)
		work.Set(0, d.Get(0)+tau+d.Get(1))
		work.Set(1, two*d.Get(1)+tau)
		//        DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
		//        DELTA( 2 ) = -Z( 2 ) / TAU
		//        TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
		//        DELTA( 1 ) = DELTA( 1 ) / TEMP
		//        DELTA( 2 ) = DELTA( 2 ) / TEMP
	}
}
