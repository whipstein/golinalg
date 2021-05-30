package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
	"math"
)

// Dlaic1 applies one step of incremental condition estimation in
// its simplest version:
//
// Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j
// lower triangular matrix L, such that
//          twonorm(L*x) = sest
// Then DLAIC1 computes sestpr, s, c such that
// the vector
//                 [ s*x ]
//          xhat = [  c  ]
// is an approximate singular vector of
//                 [ L       0  ]
//          Lhat = [ w**T gamma ]
// in the sense that
//          twonorm(Lhat*xhat) = sestpr.
//
// Depending on JOB, an estimate for the largest or smallest singular
// value is computed.
//
// Note that [s c]**T and sestpr**2 is an eigenpair of the system
//
//     diag(sest*sest, 0) + [alpha  gamma] * [ alpha ]
//                                           [ gamma ]
//
// where  alpha =  x**T*w.
func Dlaic1(job, j *int, x *mat.Vector, sest *float64, w *mat.Vector, gamma, sestpr, s, c *float64) {
	var absalp, absest, absgam, alpha, b, cosine, eps, four, half, norma, one, s1, s2, sine, t, test, tmp, two, zero, zeta1, zeta2 float64

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	four = 4.0

	eps = Dlamch(Epsilon)
	alpha = goblas.Ddot(j, x, toPtr(1), w, toPtr(1))

	absalp = math.Abs(alpha)
	absgam = math.Abs(*gamma)
	absest = math.Abs(*sest)

	if (*job) == 1 {
		//        Estimating largest singular value
		//
		//        special cases
		if (*sest) == zero {
			s1 = maxf64(absgam, absalp)
			if s1 == zero {
				(*s) = zero
				(*c) = one
				(*sestpr) = zero
			} else {
				(*s) = alpha / s1
				(*c) = (*gamma) / s1
				tmp = math.Sqrt((*s)*(*s) + (*c)*(*c))
				(*s) = (*s) / tmp
				(*c) = (*c) / tmp
				(*sestpr) = s1 * tmp
			}
			return
		} else if absgam <= eps*absest {
			(*s) = one
			(*c) = zero
			tmp = maxf64(absest, absalp)
			s1 = absest / tmp
			s2 = absalp / tmp
			(*sestpr) = tmp * math.Sqrt(s1*s1+s2*s2)
			return
		} else if absalp <= eps*absest {
			s1 = absgam
			s2 = absest
			if s1 <= s2 {
				(*s) = one
				(*c) = zero
				(*sestpr) = s2
			} else {
				(*s) = zero
				(*c) = one
				(*sestpr) = s1
			}
			return
		} else if absest <= eps*absalp || absest <= eps*absgam {
			s1 = absgam
			s2 = absalp
			if s1 <= s2 {
				tmp = s1 / s2
				(*s) = math.Sqrt(one + tmp*tmp)
				(*sestpr) = s2 * (*s)
				(*c) = ((*gamma) / s2) / (*s)
				(*s) = signf64(one, alpha) / (*s)
			} else {
				tmp = s2 / s1
				(*c) = math.Sqrt(one + tmp*tmp)
				(*sestpr) = s1 * (*c)
				(*s) = (alpha / s1) / (*c)
				(*c) = signf64(one, *gamma) / (*c)
			}
			return
		} else {
			//           normal case
			zeta1 = alpha / absest
			zeta2 = (*gamma) / absest

			b = (one - zeta1*zeta1 - zeta2*zeta2) * half
			(*c) = zeta1 * zeta1
			if b > zero {
				t = (*c) / (b + math.Sqrt(b*b+(*c)))
			} else {
				t = math.Sqrt(b*b+(*c)) - b
			}

			sine = -zeta1 / t
			cosine = -zeta2 / (one + t)
			tmp = math.Sqrt(sine*sine + cosine*cosine)
			(*s) = sine / tmp
			(*c) = cosine / tmp
			(*sestpr) = math.Sqrt(t+one) * absest
			return
		}

	} else if (*job) == 2 {
		//        Estimating smallest singular value
		//
		//        special cases
		if (*sest) == zero {
			(*sestpr) = zero
			if maxf64(absgam, absalp) == zero {
				sine = one
				cosine = zero
			} else {
				sine = -(*gamma)
				cosine = alpha
			}
			s1 = maxf64(math.Abs(sine), math.Abs(cosine))
			(*s) = sine / s1
			(*c) = cosine / s1
			tmp = math.Sqrt((*s)*(*s) + (*c)*(*c))
			(*s) = (*s) / tmp
			(*c) = (*c) / tmp
			return
		} else if absgam <= eps*absest {
			(*s) = zero
			(*c) = one
			(*sestpr) = absgam
			return
		} else if absalp <= eps*absest {
			s1 = absgam
			s2 = absest
			if s1 <= s2 {
				(*s) = zero
				(*c) = one
				(*sestpr) = s1
			} else {
				(*s) = one
				(*c) = zero
				(*sestpr) = s2
			}
			return
		} else if absest <= eps*absalp || absest <= eps*absgam {
			s1 = absgam
			s2 = absalp
			if s1 <= s2 {
				tmp = s1 / s2
				(*c) = math.Sqrt(one + tmp*tmp)
				(*sestpr) = absest * (tmp / (*c))
				(*s) = -((*gamma) / s2) / (*c)
				(*c) = signf64(one, alpha) / (*c)
			} else {
				tmp = s2 / s1
				(*s) = math.Sqrt(one + tmp*tmp)
				(*sestpr) = absest / (*s)
				(*c) = (alpha / s1) / (*s)
				(*s) = -signf64(one, *gamma) / (*s)
			}
			return
		} else {
			//           normal case
			zeta1 = alpha / absest
			zeta2 = (*gamma) / absest

			norma = maxf64(one+zeta1*zeta1+math.Abs(zeta1*zeta2), math.Abs(zeta1*zeta2)+zeta2*zeta2)

			//           See if root is closer to zero or to ONE
			test = one + two*(zeta1-zeta2)*(zeta1+zeta2)
			if test >= zero {
				//              root is close to zero, compute directly
				b = (zeta1*zeta1 + zeta2*zeta2 + one) * half
				(*c) = zeta2 * zeta2
				t = (*c) / (b + math.Sqrt(math.Abs(b*b-(*c))))
				sine = zeta1 / (one - t)
				cosine = -zeta2 / t
				(*sestpr) = math.Sqrt(t+four*eps*eps*norma) * absest
			} else {
				//              root is closer to ONE, shift by that amount
				b = (zeta2*zeta2 + zeta1*zeta1 - one) * half
				(*c) = zeta1 * zeta1
				if b >= zero {
					t = -(*c) / (b + math.Sqrt(b*b+(*c)))
				} else {
					t = b - math.Sqrt(b*b+(*c))
				}
				sine = -zeta1 / t
				cosine = -zeta2 / (one + t)
				(*sestpr) = math.Sqrt(one+t+four*eps*eps*norma) * absest
			}
			tmp = math.Sqrt(sine*sine + cosine*cosine)
			(*s) = sine / tmp
			(*c) = cosine / tmp
			return

		}
	}
}
