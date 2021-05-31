package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlar1v computes the (scaled) r-th column of the inverse of
// the sumbmatrix in rows B1 through BN of the tridiagonal matrix
// L D L**T - sigma I. When sigma is close to an eigenvalue, the
// computed vector is an accurate eigenvector. Usually, r corresponds
// to the index where the eigenvector is largest in magnitude.
// The following steps accomplish this computation :
// (a) Stationary qd transform,  L D L**T - sigma I = L(+) D(+) L(+)**T,
// (b) Progressive qd transform, L D L**T - sigma I = U(-) D(-) U(-)**T,
// (c) Computation of the diagonal elements of the inverse of
//     L D L**T - sigma I by combining the above transforms, and choosing
//     r as the index where the diagonal of the inverse is (one of the)
//     largest in magnitude.
// (d) Computation of the (scaled) r-th column of the inverse using the
//     twisted factorization obtained by combining the top part of the
//     the stationary and the bottom part of the progressive transform.
func Zlar1v(n, b1, bn *int, lambda *float64, d, l, ld, lld *mat.Vector, pivmin, gaptol *float64, z *mat.CVector, wantnc bool, negcnt *int, ztz, mingma *float64, r *int, isuppz *[]int, nrminv, resid, rqcorr *float64, work *mat.Vector) {
	var sawnan1, sawnan2 bool
	var cone complex128
	var dminus, dplus, eps, one, s, tmp, zero float64
	var i, indlpl, indp, inds, indumn, neg1, neg2, r1, r2 int

	zero = 0.0
	one = 1.0
	cone = (1.0 + 0.0*1i)

	eps = Dlamch(Precision)
	if (*r) == 0 {
		r1 = (*b1)
		r2 = (*bn)
	} else {
		r1 = (*r)
		r2 = (*r)
	}
	//     Storage for LPLUS
	indlpl = 0
	//     Storage for UMINUS
	indumn = (*n)
	inds = 2*(*n) + 1
	indp = 3*(*n) + 1
	if (*b1) == 1 {
		work.Set(inds-1, zero)
	} else {
		work.Set(inds+(*b1)-1-1, lld.Get((*b1)-1-1))
	}

	//     Compute the stationary transform (using the differential form)
	//     until the index R2.
	sawnan1 = false
	neg1 = 0
	s = work.Get(inds+(*b1)-1-1) - (*lambda)
	for i = (*b1); i <= r1-1; i++ {
		dplus = d.Get(i-1) + s
		work.Set(indlpl+i-1, ld.Get(i-1)/dplus)
		if dplus < zero {
			neg1 = neg1 + 1
		}
		work.Set(inds+i-1, s*work.Get(indlpl+i-1)*l.Get(i-1))
		s = work.Get(inds+i-1) - (*lambda)
	}
	sawnan1 = Disnan(int(s))
	if sawnan1 {
		goto label60
	}
	for i = r1; i <= r2-1; i++ {
		dplus = d.Get(i-1) + s
		work.Set(indlpl+i-1, ld.Get(i-1)/dplus)
		work.Set(inds+i-1, s*work.Get(indlpl+i-1)*l.Get(i-1))
		s = work.Get(inds+i-1) - (*lambda)
	}
	sawnan1 = Disnan(int(s))

label60:
	;
	if sawnan1 {
		//        Runs a slower version of the above loop if a NaN is detected
		neg1 = 0
		s = work.Get(inds+(*b1)-1-1) - (*lambda)
		for i = (*b1); i <= r1-1; i++ {
			dplus = d.Get(i-1) + s
			if math.Abs(dplus) < (*pivmin) {
				dplus = -(*pivmin)
			}
			work.Set(indlpl+i-1, ld.Get(i-1)/dplus)
			if dplus < zero {
				neg1 = neg1 + 1
			}
			work.Set(inds+i-1, s*work.Get(indlpl+i-1)*l.Get(i-1))
			if work.Get(indlpl+i-1) == zero {
				work.Set(inds+i-1, lld.Get(i-1))
			}
			s = work.Get(inds+i-1) - (*lambda)
		}
		for i = r1; i <= r2-1; i++ {
			dplus = d.Get(i-1) + s
			if math.Abs(dplus) < (*pivmin) {
				dplus = -(*pivmin)
			}
			work.Set(indlpl+i-1, ld.Get(i-1)/dplus)
			work.Set(inds+i-1, s*work.Get(indlpl+i-1)*l.Get(i-1))
			if work.Get(indlpl+i-1) == zero {
				work.Set(inds+i-1, lld.Get(i-1))
			}
			s = work.Get(inds+i-1) - (*lambda)
		}
	}

	//     Compute the progressive transform (using the differential form)
	//     until the index R1
	sawnan2 = false
	neg2 = 0
	work.Set(indp+(*bn)-1-1, d.Get((*bn)-1)-(*lambda))
	for i = (*bn) - 1; i >= r1; i -= 1 {
		dminus = lld.Get(i-1) + work.Get(indp+i-1)
		tmp = d.Get(i-1) / dminus
		if dminus < zero {
			neg2 = neg2 + 1
		}
		work.Set(indumn+i-1, l.Get(i-1)*tmp)
		work.Set(indp+i-1-1, work.Get(indp+i-1)*tmp-(*lambda))
	}
	tmp = work.Get(indp + r1 - 1 - 1)
	sawnan2 = Disnan(int(tmp))
	if sawnan2 {
		//        Runs a slower version of the above loop if a NaN is detected
		neg2 = 0
		for i = (*bn) - 1; i >= r1; i-- {
			dminus = lld.Get(i-1) + work.Get(indp+i-1)
			if math.Abs(dminus) < (*pivmin) {
				dminus = -(*pivmin)
			}
			tmp = d.Get(i-1) / dminus
			if dminus < zero {
				neg2 = neg2 + 1
			}
			work.Set(indumn+i-1, l.Get(i-1)*tmp)
			work.Set(indp+i-1-1, work.Get(indp+i-1)*tmp-(*lambda))
			if tmp == zero {
				work.Set(indp+i-1-1, d.Get(i-1)-(*lambda))
			}
		}
	}

	//     Find the index (from R1 to R2) of the largest (in magnitude)
	//     diagonal element of the inverse
	(*mingma) = work.Get(inds+r1-1-1) + work.Get(indp+r1-1-1)
	if (*mingma) < zero {
		neg1 = neg1 + 1
	}
	if wantnc {
		(*negcnt) = neg1 + neg2
	} else {
		(*negcnt) = -1
	}
	if math.Abs(*mingma) == zero {
		(*mingma) = eps * work.Get(inds+r1-1-1)
	}
	(*r) = r1
	for i = r1; i <= r2-1; i++ {
		tmp = work.Get(inds+i-1) + work.Get(indp+i-1)
		if tmp == zero {
			tmp = eps * work.Get(inds+i-1)
		}
		if math.Abs(tmp) <= math.Abs(*mingma) {
			(*mingma) = tmp
			(*r) = i + 1
		}
	}

	//     Compute the FP vector: solve N^T v = e_r
	(*isuppz)[0] = (*b1)
	(*isuppz)[1] = (*bn)
	z.Set((*r)-1, cone)
	(*ztz) = one

	//     Compute the FP vector upwards from R
	if !sawnan1 && !sawnan2 {
		for i = (*r) - 1; i >= (*b1); i -= 1 {
			z.Set(i-1, -(work.GetCmplx(indlpl+i-1) * z.Get(i+1-1)))
			if (z.GetMag(i-1)+z.GetMag(i+1-1))*ld.GetMag(i-1) < (*gaptol) {
				z.SetRe(i-1, zero)
				(*isuppz)[0] = i + 1
				goto label220
			}
			(*ztz) = (*ztz) + real(z.Get(i-1)*z.Get(i-1))
		}
	label220:
	} else {
		//        Run slower loop if NaN occurred.
		for i = (*r) - 1; i >= (*b1); i-- {
			if z.GetRe(i+1-1) == zero {
				z.Set(i-1, -(ld.GetCmplx(i+1-1)/ld.GetCmplx(i-1))*z.Get(i+2-1))
			} else {
				z.Set(i-1, -(work.GetCmplx(indlpl+i-1) * z.Get(i+1-1)))
			}
			if (z.GetMag(i-1)+z.GetMag(i+1-1))*ld.GetMag(i-1) < (*gaptol) {
				z.SetRe(i-1, zero)
				(*isuppz)[0] = i + 1
				goto label240
			}
			(*ztz) = (*ztz) + real(z.Get(i-1)*z.Get(i-1))
		}
	label240:
	}
	//     Compute the FP vector downwards from R in blocks of size BLKSIZ
	if !sawnan1 && !sawnan2 {
		for i = (*r); i <= (*bn)-1; i++ {
			z.Set(i+1-1, -(work.GetCmplx(indumn+i-1) * z.Get(i-1)))
			if (z.GetMag(i-1)+z.GetMag(i+1-1))*ld.GetMag(i-1) < (*gaptol) {
				z.SetRe(i+1-1, zero)
				(*isuppz)[1] = i
				goto label260
			}
			(*ztz) = (*ztz) + real(z.Get(i+1-1)*z.Get(i+1-1))
		}
	label260:
	} else {
		//        Run slower loop if NaN occurred.
		for i = (*r); i <= (*bn)-1; i++ {
			if z.GetRe(i-1) == zero {
				z.Set(i+1-1, -(ld.GetCmplx(i-1-1)/ld.GetCmplx(i-1))*z.Get(i-1-1))
			} else {
				z.Set(i+1-1, -(work.GetCmplx(indumn+i-1) * z.Get(i-1)))
			}
			if (z.GetMag(i-1)+z.GetMag(i+1-1))*ld.GetMag(i-1) < (*gaptol) {
				z.SetRe(i+1-1, zero)
				(*isuppz)[1] = i
				goto label280
			}
			(*ztz) = (*ztz) + real(z.Get(i+1-1)*z.Get(i+1-1))
		}
	label280:
	}

	//     Compute quantities for convergence test
	tmp = one / (*ztz)
	(*nrminv) = math.Sqrt(tmp)
	(*resid) = math.Abs(*mingma) * (*nrminv)
	(*rqcorr) = (*mingma) * tmp
}
