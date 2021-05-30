package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dstebz computes the eigenvalues of a symmetric tridiagonal
// matrix T.  The user may ask for all eigenvalues, all eigenvalues
// in the half-open interval (VL, VU], or the IL-th through IU-th
// eigenvalues.
//
// To avoid overflow, the matrix must be scaled so that its
// largest element is no greater than overflow**(1/2) * underflow**(1/4) in absolute value, and for greatest
// accuracy, it should not be much smaller than that.
//
// See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
// Matrix", Report CS41, Computer Science Dept., Stanford
// University, July 21, 1966.
func Dstebz(_range, order byte, n *int, vl, vu *float64, il, iu *int, abstol *float64, d, e *mat.Vector, m, nsplit *int, w *mat.Vector, iblock, isplit *[]int, work *mat.Vector, iwork *[]int, info *int) {
	var ncnvrg, toofew bool
	var atoli, bnorm, fudge, gl, gu, half, one, pivmin, relfac, rtoli, safemn, tmp1, tmp2, tnorm, two, ulp, wkill, wl, wlu, wu, wul, zero float64
	var ib, ibegin, idiscl, idiscu, ie, iend, iinfo, im, in, ioff, iorder, iout, irange, itmax, itmp1, iw, iwoff, j, jb, jdisc, je, nb, nwl, nwu int
	idumma := make([]int, 1)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = 1.0 / two
	fudge = 2.1
	relfac = 2.0

	(*info) = 0

	//     Decode RANGE
	if _range == 'A' {
		irange = 1
	} else if _range == 'V' {
		irange = 2
	} else if _range == 'I' {
		irange = 3
	} else {
		irange = 0
	}

	//     Decode ORDER
	if order == 'B' {
		iorder = 2
	} else if order == 'E' {
		iorder = 1
	} else {
		iorder = 0
	}

	//     Check for Errors
	if irange <= 0 {
		(*info) = -1
	} else if iorder <= 0 {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if irange == 2 {
		if (*vl) >= (*vu) {
			(*info) = -5
		}
	} else if irange == 3 && ((*il) < 1 || (*il) > maxint(1, *n)) {
		(*info) = -6
	} else if irange == 3 && ((*iu) < minint(*n, *il) || (*iu) > (*n)) {
		(*info) = -7
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("DSTEBZ"), -(*info))
		return
	}

	//     Initialize error flags
	(*info) = 0
	ncnvrg = false
	toofew = false

	//     Quick return if possible
	(*m) = 0
	if (*n) == 0 {
		return
	}

	//     Simplifications:
	if irange == 3 && (*il) == 1 && (*iu) == (*n) {
		irange = 1
	}

	//     Get machine constants
	//     NB is the minimum vector length for vector bisection, or 0
	//     if only scalar is to be done.
	safemn = Dlamch(SafeMinimum)
	ulp = Dlamch(Precision)
	rtoli = ulp * relfac
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSTEBZ"), []byte{' '}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nb <= 1 {
		nb = 0
	}

	//     Special Case when N=1
	if (*n) == 1 {
		(*nsplit) = 1
		(*isplit)[0] = 1
		if irange == 2 && ((*vl) >= d.Get(0) || (*vu) < d.Get(0)) {
			(*m) = 0
		} else {
			w.Set(0, d.Get(0))
			(*iblock)[0] = 1
			(*m) = 1
		}
		return
	}

	//     Compute Splitting Points
	(*nsplit) = 1
	work.Set((*n)-1, zero)
	pivmin = one

	for j = 2; j <= (*n); j++ {
		tmp1 = math.Pow(e.Get(j-1-1), 2)
		if math.Abs(d.Get(j-1)*d.Get(j-1-1))*math.Pow(ulp, 2)+safemn > tmp1 {
			(*isplit)[(*nsplit)-1] = j - 1
			(*nsplit) = (*nsplit) + 1
			work.Set(j-1-1, zero)
		} else {
			work.Set(j-1-1, tmp1)
			pivmin = maxf64(pivmin, tmp1)
		}
	}
	(*isplit)[(*nsplit)-1] = (*n)
	pivmin = pivmin * safemn

	//     Compute Interval and ATOLI
	if irange == 3 {
		//        RANGE='I': Compute the interval containing eigenvalues
		//                   IL through IU.
		//
		//        Compute Gershgorin interval for entire (split) matrix
		//        and use it as the initial interval
		gu = d.Get(0)
		gl = d.Get(0)
		tmp1 = zero

		for j = 1; j <= (*n)-1; j++ {
			tmp2 = math.Sqrt(work.Get(j - 1))
			gu = maxf64(gu, d.Get(j-1)+tmp1+tmp2)
			gl = minf64(gl, d.Get(j-1)-tmp1-tmp2)
			tmp1 = tmp2
		}

		gu = maxf64(gu, d.Get((*n)-1)+tmp1)
		gl = minf64(gl, d.Get((*n)-1)-tmp1)
		tnorm = maxf64(math.Abs(gl), math.Abs(gu))
		gl = gl - fudge*tnorm*ulp*float64(*n) - fudge*two*pivmin
		gu = gu + fudge*tnorm*ulp*float64(*n) + fudge*pivmin

		//        Compute Iteration parameters
		itmax = int((math.Log(tnorm+pivmin)-math.Log(pivmin))/math.Log(two)) + 2
		if (*abstol) <= zero {
			atoli = ulp * tnorm
		} else {
			atoli = (*abstol)
		}

		work.Set((*n)+1-1, gl)
		work.Set((*n)+2-1, gl)
		work.Set((*n)+3-1, gu)
		work.Set((*n)+4-1, gu)
		work.Set((*n)+5-1, gl)
		work.Set((*n)+6-1, gu)
		(*iwork)[0] = -1
		(*iwork)[1] = -1
		(*iwork)[2] = (*n) + 1
		(*iwork)[3] = (*n) + 1
		(*iwork)[4] = (*il) - 1
		(*iwork)[5] = (*iu)

		Dlaebz(func() *int { y := 3; return &y }(), &itmax, n, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &nb, &atoli, &rtoli, &pivmin, d, e, work, toSlice(iwork, 4), work.MatrixOff((*n)+1-1, 2, opts), work.Off((*n)+5-1), &iout, iwork, w, iblock, &iinfo)

		if (*iwork)[5] == (*iu) {
			wl = work.Get((*n) + 1 - 1)
			wlu = work.Get((*n) + 3 - 1)
			nwl = (*iwork)[0]
			wu = work.Get((*n) + 4 - 1)
			wul = work.Get((*n) + 2 - 1)
			nwu = (*iwork)[3]
		} else {
			wl = work.Get((*n) + 2 - 1)
			wlu = work.Get((*n) + 4 - 1)
			nwl = (*iwork)[1]
			wu = work.Get((*n) + 3 - 1)
			wul = work.Get((*n) + 1 - 1)
			nwu = (*iwork)[2]
		}

		if nwl < 0 || nwl >= (*n) || nwu < 1 || nwu > (*n) {
			(*info) = 4
			return
		}
	} else {
		//        RANGE='A' or 'V' -- Set ATOLI
		tnorm = maxf64(math.Abs(d.Get(0))+math.Abs(e.Get(0)), math.Abs(d.Get((*n)-1))+math.Abs(e.Get((*n)-1-1)))

		for j = 2; j <= (*n)-1; j++ {
			tnorm = maxf64(tnorm, math.Abs(d.Get(j-1))+math.Abs(e.Get(j-1-1))+math.Abs(e.Get(j-1)))
		}

		if (*abstol) <= zero {
			atoli = ulp * tnorm
		} else {
			atoli = (*abstol)
		}

		if irange == 2 {
			wl = (*vl)
			wu = (*vu)
		} else {
			wl = zero
			wu = zero
		}
	}

	//     Find Eigenvalues -- Loop Over Blocks and recompute NWL and NWU.
	//     NWL accumulates the number of eigenvalues .le. WL,
	//     NWU accumulates the number of eigenvalues .le. WU
	(*m) = 0
	iend = 0
	(*info) = 0
	nwl = 0
	nwu = 0

	for jb = 1; jb <= (*nsplit); jb++ {
		ioff = iend
		ibegin = ioff + 1
		iend = (*isplit)[jb-1]
		in = iend - ioff

		if in == 1 {
			//           Special Case -- IN=1
			if irange == 1 || wl >= d.Get(ibegin-1)-pivmin {
				nwl = nwl + 1
			}
			if irange == 1 || wu >= d.Get(ibegin-1)-pivmin {
				nwu = nwu + 1
			}
			if irange == 1 || (wl < d.Get(ibegin-1)-pivmin && wu >= d.Get(ibegin-1)-pivmin) {
				(*m) = (*m) + 1
				w.Set((*m)-1, d.Get(ibegin-1))
				(*iblock)[(*m)-1] = jb
			}
		} else {
			//           General Case -- IN > 1
			//
			//           Compute Gershgorin Interval
			//           and use it as the initial interval
			gu = d.Get(ibegin - 1)
			gl = d.Get(ibegin - 1)
			tmp1 = zero

			for j = ibegin; j <= iend-1; j++ {
				tmp2 = math.Abs(e.Get(j - 1))
				gu = maxf64(gu, d.Get(j-1)+tmp1+tmp2)
				gl = minf64(gl, d.Get(j-1)-tmp1-tmp2)
				tmp1 = tmp2
			}

			gu = maxf64(gu, d.Get(iend-1)+tmp1)
			gl = minf64(gl, d.Get(iend-1)-tmp1)
			bnorm = maxf64(math.Abs(gl), math.Abs(gu))
			gl = gl - fudge*bnorm*ulp*float64(in) - fudge*pivmin
			gu = gu + fudge*bnorm*ulp*float64(in) + fudge*pivmin

			//           Compute ATOLI for the current submatrix
			if (*abstol) <= zero {
				atoli = ulp * maxf64(math.Abs(gl), math.Abs(gu))
			} else {
				atoli = (*abstol)
			}

			if irange > 1 {
				if gu < wl {
					nwl = nwl + in
					nwu = nwu + in
					goto label70
				}
				gl = maxf64(gl, wl)
				gu = minf64(gu, wu)
				if gl >= gu {
					goto label70
				}
			}

			//           Set Up Initial Interval
			work.Set((*n)+1-1, gl)
			work.Set((*n)+in+1-1, gu)
			Dlaebz(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &in, &in, func() *int { y := 1; return &y }(), &nb, &atoli, &rtoli, &pivmin, d.Off(ibegin-1), e.Off(ibegin-1), work.Off(ibegin-1), &idumma, work.MatrixOff((*n)+1-1, in, opts), work.Off((*n)+2*in+1-1), &im, iwork, w.Off((*m)+1-1), toSlice(iblock, (*m)+1-1), &iinfo)

			nwl = nwl + (*iwork)[0]
			nwu = nwu + (*iwork)[in+1-1]
			iwoff = (*m) - (*iwork)[0]

			//           Compute Eigenvalues
			itmax = int((math.Log(gu-gl+pivmin)-math.Log(pivmin))/math.Log(two)) + 2
			Dlaebz(func() *int { y := 2; return &y }(), &itmax, &in, &in, func() *int { y := 1; return &y }(), &nb, &atoli, &rtoli, &pivmin, d.Off(ibegin-1), e.Off(ibegin-1), work.Off(ibegin-1), &idumma, work.MatrixOff((*n)+1-1, in, opts), work.Off((*n)+2*in+1-1), &iout, iwork, w.Off((*m)+1-1), toSlice(iblock, (*m)+1-1), &iinfo)

			//           Copy Eigenvalues Into W and IBLOCK
			//           Use -JB for block number for unconverged eigenvalues.
			for j = 1; j <= iout; j++ {
				tmp1 = half * (work.Get(j+(*n)-1) + work.Get(j+in+(*n)-1))

				//              Flag non-convergence.
				if j > iout-iinfo {
					ncnvrg = true
					ib = -jb
				} else {
					ib = jb
				}
				for je = (*iwork)[j-1] + 1 + iwoff; je <= (*iwork)[j+in-1]+iwoff; je++ {
					w.Set(je-1, tmp1)
					(*iblock)[je-1] = ib
				}
			}

			(*m) = (*m) + im
		}
	label70:
	}

	//     If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
	//     If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
	if irange == 3 {
		im = 0
		idiscl = (*il) - 1 - nwl
		idiscu = nwu - (*iu)

		if idiscl > 0 || idiscu > 0 {
			for je = 1; je <= (*m); je++ {
				if w.Get(je-1) <= wlu && idiscl > 0 {
					idiscl = idiscl - 1
				} else if w.Get(je-1) >= wul && idiscu > 0 {
					idiscu = idiscu - 1
				} else {
					im = im + 1
					w.Set(im-1, w.Get(je-1))
					(*iblock)[im-1] = (*iblock)[je-1]
				}
			}
			(*m) = im
		}
		if idiscl > 0 || idiscu > 0 {
			//           Code to deal with effects of bad arithmetic:
			//           Some low eigenvalues to be discarded are not in (WL,WLU],
			//           or high eigenvalues to be discarded are not in (WUL,WU]
			//           so just kill off the smallest IDISCL/largest IDISCU
			//           eigenvalues, by simply finding the smallest/largest
			//           eigenvalue(s).
			//
			//           (If N(w) is monotone non-decreasing, this should never
			//               happen.)
			if idiscl > 0 {
				wkill = wu
				for jdisc = 1; jdisc <= idiscl; jdisc++ {
					iw = 0
					for je = 1; je <= (*m); je++ {
						if (*iblock)[je-1] != 0 && (w.Get(je-1) < wkill || iw == 0) {
							iw = je
							wkill = w.Get(je - 1)
						}
					}
					(*iblock)[iw-1] = 0
				}
			}
			if idiscu > 0 {

				wkill = wl
				for jdisc = 1; jdisc <= idiscu; jdisc++ {
					iw = 0
					for je = 1; je <= (*m); je++ {
						if (*iblock)[je-1] != 0 && (w.Get(je-1) > wkill || iw == 0) {
							iw = je
							wkill = w.Get(je - 1)
						}
					}
					(*iblock)[iw-1] = 0
				}
			}
			im = 0
			for je = 1; je <= (*m); je++ {
				if (*iblock)[je-1] != 0 {
					im = im + 1
					w.Set(im-1, w.Get(je-1))
					(*iblock)[im-1] = (*iblock)[je-1]
				}
			}
			(*m) = im
		}
		if idiscl < 0 || idiscu < 0 {
			toofew = true
		}
	}

	//     If ORDER='B', do nothing -- the eigenvalues are already sorted
	//        by block.
	//     If ORDER='E', sort the eigenvalues from smallest to largest
	if iorder == 1 && (*nsplit) > 1 {
		for je = 1; je <= (*m)-1; je++ {
			ie = 0
			tmp1 = w.Get(je - 1)
			for j = je + 1; j <= (*m); j++ {
				if w.Get(j-1) < tmp1 {
					ie = j
					tmp1 = w.Get(j - 1)
				}
			}

			if ie != 0 {
				itmp1 = (*iblock)[ie-1]
				w.Set(ie-1, w.Get(je-1))
				(*iblock)[ie-1] = (*iblock)[je-1]
				w.Set(je-1, tmp1)
				(*iblock)[je-1] = itmp1
			}
		}
	}

	(*info) = 0
	if ncnvrg {
		(*info) = (*info) + 1
	}
	if toofew {
		(*info) = (*info) + 2
	}
}
