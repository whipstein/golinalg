package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlaic1 applies one step of incremental condition estimation in
// its simplest version:
//
// Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j
// lower triangular matrix L, such that
//          twonorm(L*x) = sest
// Then ZLAIC1 computes sestpr, s, c such that
// the vector
//                 [ s*x ]
//          xhat = [  c  ]
// is an approximate singular vector of
//                 [ L       0  ]
//          Lhat = [ w**H gamma ]
// in the sense that
//          twonorm(Lhat*xhat) = sestpr.
//
// Depending on JOB, an estimate for the largest or smallest singular
// value is computed.
//
// Note that [s c]**H and sestpr**2 is an eigenpair of the system
//
//     diag(sest*sest, 0) + [alpha  gamma] * [ conjg(alpha) ]
//                                           [ conjg(gamma) ]
//
// where  alpha =  x**H * w.
func Zlaic1(job, j int, x *mat.CVector, sest float64, w *mat.CVector, gamma complex128) (sestpr float64, s, c complex128) {
	var alpha, cosine, sine complex128
	var absalp, absest, absgam, b, eps, four, half, norma, one, s1, s2, scl, t, test, tmp, two, zero, zeta1, zeta2 float64

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 0.5
	four = 4.0

	eps = Dlamch(Epsilon)
	alpha = w.Dotc(j, x, 1, 1)

	absalp = cmplx.Abs(alpha)
	absgam = cmplx.Abs(gamma)
	absest = math.Abs(sest)

	if job == 1 {
		//        Estimating largest singular value
		//
		//        special cases
		if sest == zero {
			s1 = math.Max(absgam, absalp)
			if s1 == zero {
				s = complex(zero, 0)
				c = complex(one, 0)
				sestpr = zero
			} else {
				s = alpha / complex(s1, 0)
				c = gamma / complex(s1, 0)
				tmp = math.Sqrt(real(s*cmplx.Conj(s)) + real(c*cmplx.Conj(c)))
				s = s / complex(tmp, 0)
				c = c / complex(tmp, 0)
				sestpr = s1 * tmp
			}
			return
		} else if absgam <= eps*absest {
			s = complex(one, 0)
			c = complex(zero, 0)
			tmp = math.Max(absest, absalp)
			s1 = absest / tmp
			s2 = absalp / tmp
			sestpr = tmp * math.Sqrt(s1*s1+s2*s2)
			return
		} else if absalp <= eps*absest {
			s1 = absgam
			s2 = absest
			if s1 <= s2 {
				s = complex(one, 0)
				c = complex(zero, 0)
				sestpr = s2
			} else {
				s = complex(zero, 0)
				c = complex(one, 0)
				sestpr = s1
			}
			return
		} else if absest <= eps*absalp || absest <= eps*absgam {
			s1 = absgam
			s2 = absalp
			if s1 <= s2 {
				tmp = s1 / s2
				scl = math.Sqrt(one + tmp*tmp)
				sestpr = s2 * scl
				s = (alpha / complex(s2, 0)) / complex(scl, 0)
				c = (gamma / complex(s2, 0)) / complex(scl, 0)
			} else {
				tmp = s2 / s1
				scl = math.Sqrt(one + tmp*tmp)
				sestpr = s1 * scl
				s = (alpha / complex(s1, 0)) / complex(scl, 0)
				c = (gamma / complex(s1, 0)) / complex(scl, 0)
			}
			return
		} else {
			//           normal case
			zeta1 = absalp / absest
			zeta2 = absgam / absest

			b = (one - zeta1*zeta1 - zeta2*zeta2) * half
			c = complex(zeta1*zeta1, 0)
			if b > zero {
				t = real(c) / (b + math.Sqrt(b*b+real(c)))
			} else {
				t = math.Sqrt(b*b+real(c)) - b
			}

			sine = -(alpha / complex(absest, 0)) / complex(t, 0)
			cosine = -(gamma / complex(absest, 0)) / complex(one+t, 0)
			tmp = math.Sqrt(real(sine*cmplx.Conj(sine)) + real(cosine*cmplx.Conj(cosine)))
			s = sine / complex(tmp, 0)
			c = cosine / complex(tmp, 0)
			sestpr = math.Sqrt(t+one) * absest
			return
		}

	} else if job == 2 {
		//        Estimating smallest singular value
		//
		//        special cases
		if sest == zero {
			sestpr = zero
			if math.Max(absgam, absalp) == zero {
				sine = complex(one, 0)
				cosine = complex(zero, 0)
			} else {
				sine = -cmplx.Conj(gamma)
				cosine = cmplx.Conj(alpha)
			}
			s1 = math.Max(math.Abs(real(sine)), math.Abs(real(cosine)))
			s = sine / complex(s1, 0)
			c = cosine / complex(s1, 0)
			tmp = math.Sqrt(real(s*cmplx.Conj(s)) + real(c*cmplx.Conj(c)))
			s = s / complex(tmp, 0)
			c = c / complex(tmp, 0)
			return
		} else if absgam <= eps*absest {
			s = complex(zero, 0)
			c = complex(one, 0)
			sestpr = absgam
			return
		} else if absalp <= eps*absest {
			s1 = absgam
			s2 = absest
			if s1 <= s2 {
				s = complex(zero, 0)
				c = complex(one, 0)
				sestpr = s1
			} else {
				s = complex(one, 0)
				c = complex(zero, 0)
				sestpr = s2
			}
			return
		} else if absest <= eps*absalp || absest <= eps*absgam {
			s1 = absgam
			s2 = absalp
			if s1 <= s2 {
				tmp = s1 / s2
				scl = math.Sqrt(one + tmp*tmp)
				sestpr = absest * (tmp / scl)
				s = -(cmplx.Conj(gamma) / complex(s2, 0)) / complex(scl, 0)
				c = (cmplx.Conj(alpha) / complex(s2, 0)) / complex(scl, 0)
			} else {
				tmp = s2 / s1
				scl = math.Sqrt(one + tmp*tmp)
				sestpr = absest / scl
				s = -(cmplx.Conj(gamma) / complex(s1, 0)) / complex(scl, 0)
				c = (cmplx.Conj(alpha) / complex(s1, 0)) / complex(scl, 0)
			}
			return
		} else {
			//           normal case
			zeta1 = absalp / absest
			zeta2 = absgam / absest

			norma = math.Max(one+zeta1*zeta1+zeta1*zeta2, zeta1*zeta2+zeta2*zeta2)

			//           See if root is closer to zero or to ONE
			test = one + two*(zeta1-zeta2)*(zeta1+zeta2)
			if test >= zero {
				//              root is close to zero, compute directly
				b = (zeta1*zeta1 + zeta2*zeta2 + one) * half
				c = complex(zeta2*zeta2, 0)
				t = real(c) / (b + math.Sqrt(math.Abs(b*b-real(c))))
				sine = (alpha / complex(absest, 0)) / complex(one-t, 0)
				cosine = -(gamma / complex(absest, 0)) / complex(t, 0)
				sestpr = math.Sqrt(t+four*eps*eps*norma) * absest
			} else {
				//              root is closer to ONE, shift by that amount
				b = (zeta2*zeta2 + zeta1*zeta1 - one) * half
				c = complex(zeta1*zeta1, 0)
				if b >= zero {
					t = -real(c) / (b + math.Sqrt(b*b+real(c)))
				} else {
					t = b - math.Sqrt(b*b+real(c))
				}
				sine = -(alpha / complex(absest, 0)) / complex(t, 0)
				cosine = -(gamma / complex(absest, 0)) / complex(one+t, 0)
				sestpr = math.Sqrt(one+t+four*eps*eps*norma) * absest
			}
			tmp = math.Sqrt(real(sine*cmplx.Conj(sine)) + real(cosine*cmplx.Conj(cosine)))
			s = sine / complex(tmp, 0)
			c = cosine / complex(tmp, 0)
			return

		}
	}

	return
}
