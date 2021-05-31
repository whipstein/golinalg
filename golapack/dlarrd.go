package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Dlarrd computes the eigenvalues of a symmetric tridiagonal
// matrix T to suitable accuracy. This is an auxiliary code to be
// called from DSTEMR.
// The user may ask for all eigenvalues, all eigenvalues
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
func Dlarrd(_range, order byte, n *int, vl, vu *float64, il, iu *int, gers *mat.Vector, reltol *float64, d, e, e2 *mat.Vector, pivmin *float64, nsplit *int, isplit *[]int, m *int, w, werr *mat.Vector, wl, wu *float64, iblock, indexw *[]int, work *mat.Vector, iwork *[]int, info *int) {
	var ncnvrg, toofew bool
	var atoli, eps, fudge, gl, gu, half, one, rtoli, tmp1, tmp2, tnorm, two, uflow, wkill, wlu, wul, zero float64
	var allrng, i, ib, ibegin, idiscl, idiscu, ie, iend, iinfo, im, in, indrng, ioff, iout, irange, itmax, itmp1, itmp2, iw, iwoff, j, jblk, jdisc, je, jee, nb, nwl, nwu, valrng int
	idumma := make([]int, 1)

	zero = 0.0
	one = 1.0
	two = 2.0
	half = one / two
	fudge = two
	allrng = 1
	valrng = 2
	indrng = 3

	(*info) = 0

	//     Quick return if possible
	if (*n) <= 0 {
		return
	}

	//     Decode RANGE
	if _range == 'A' {
		irange = allrng
	} else if _range == 'V' {
		irange = valrng
	} else if _range == 'I' {
		irange = indrng
	} else {
		irange = 0
	}

	//     Check for Errors
	if irange <= 0 {
		(*info) = -1
	} else if !(order == 'B' || order == 'E') {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -3
	} else if irange == valrng {
		if (*vl) >= (*vu) {
			(*info) = -5
		}
	} else if irange == indrng && ((*il) < 1 || (*il) > maxint(1, *n)) {
		(*info) = -6
	} else if irange == indrng && ((*iu) < minint(*n, *il) || (*iu) > (*n)) {
		(*info) = -7
	}

	if (*info) != 0 {
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
	//     Simplification:
	if irange == indrng && (*il) == 1 && (*iu) == (*n) {
		irange = 1
	}
	//     Get machine constants
	eps = Dlamch(Precision)
	uflow = Dlamch(Underflow)
	//     Special Case when N=1
	//     Treat case of 1x1 matrix for quick return
	if (*n) == 1 {
		if (irange == allrng) || ((irange == valrng) && (d.Get(0) > (*vl)) && (d.Get(0) <= (*vu))) || ((irange == indrng) && ((*il) == 1) && ((*iu) == 1)) {
			(*m) = 1
			w.Set(0, d.Get(0))
			//           The computation error of the eigenvalue is zero
			werr.Set(0, zero)
			(*iblock)[0] = 1
			(*indexw)[0] = 1
		}
		return
	}
	//     NB is the minimum vector length for vector bisection, or 0
	//     if only scalar is to be done.
	nb = Ilaenv(func() *int { y := 1; return &y }(), []byte("DSTEBZ"), []byte{' '}, n, toPtr(-1), toPtr(-1), toPtr(-1))
	if nb <= 1 {
		nb = 0
	}
	//     Find global spectral radius
	gl = d.Get(0)
	gu = d.Get(0)
	for i = 1; i <= (*n); i++ {
		gl = minf64(gl, gers.Get(2*i-1-1))
		gu = maxf64(gu, gers.Get(2*i-1))
	}
	//     Compute global Gerschgorin bounds and spectral diameter
	tnorm = maxf64(math.Abs(gl), math.Abs(gu))
	gl = gl - fudge*tnorm*eps*float64(*n) - fudge*two*(*pivmin)
	gu = gu + fudge*tnorm*eps*float64(*n) + fudge*two*(*pivmin)
	//     [JAN/28/2009] remove the line below since SPDIAM variable not use
	//     SPDIAM = GU - GL
	//     Input arguments for DLAEBZ:
	//     The relative tolerance.  An interval (a,b] lies within
	//     "relative tolerance" if  b-a < RELTOL*max(|a|,|b|),
	rtoli = (*reltol)
	//     Set the absolute tolerance for interval convergence to zero to force
	//     interval convergence based on relative size of the interval.
	//     This is dangerous because intervals might not converge when RELTOL is
	//     small. But at least a very small number should be selected so that for
	//     strongly graded matrices, the code can get relatively accurate
	//     eigenvalues.
	atoli = fudge*two*uflow + fudge*two*(*pivmin)
	if irange == indrng {
		//        RANGE='I': Compute an interval containing eigenvalues
		//        IL through IU. The initial interval [GL,GU] from the global
		//        Gerschgorin bounds GL and GU is refined by DLAEBZ.
		itmax = int((math.Log(tnorm+(*pivmin))-math.Log(*pivmin))/math.Log(two)) + 2
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

		Dlaebz(func() *int { y := 3; return &y }(), &itmax, n, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &nb, &atoli, &rtoli, pivmin, d, e, e2, toSlice(iwork, 4), work.MatrixOff((*n)+1-1, 2, opts), work.Off((*n)+5-1), &iout, iwork, w, iblock, &iinfo)
		if iinfo != 0 {
			(*info) = iinfo
			return
		}
		//        On exit, output intervals may not be ordered by ascending negcount
		if (*iwork)[5] == (*iu) {
			(*wl) = work.Get((*n) + 1 - 1)
			wlu = work.Get((*n) + 3 - 1)
			nwl = (*iwork)[0]
			(*wu) = work.Get((*n) + 4 - 1)
			wul = work.Get((*n) + 2 - 1)
			nwu = (*iwork)[3]
		} else {
			(*wl) = work.Get((*n) + 2 - 1)
			wlu = work.Get((*n) + 4 - 1)
			nwl = (*iwork)[1]
			(*wu) = work.Get((*n) + 3 - 1)
			wul = work.Get((*n) + 1 - 1)
			nwu = (*iwork)[2]
		}
		//        On exit, the interval [WL, WLU] contains a value with negcount NWL,
		//        and [WUL, WU] contains a value with negcount NWU.
		if nwl < 0 || nwl >= (*n) || nwu < 1 || nwu > (*n) {
			(*info) = 4
			return
		}
	} else if irange == valrng {
		(*wl) = (*vl)
		(*wu) = (*vu)
	} else if irange == allrng {
		(*wl) = gl
		(*wu) = gu
	}
	//     Find Eigenvalues -- Loop Over blocks and recompute NWL and NWU.
	//     NWL accumulates the number of eigenvalues .le. WL,
	//     NWU accumulates the number of eigenvalues .le. WU
	(*m) = 0
	iend = 0
	(*info) = 0
	nwl = 0
	nwu = 0

	for jblk = 1; jblk <= (*nsplit); jblk++ {
		ioff = iend
		ibegin = ioff + 1
		iend = (*isplit)[jblk-1]
		in = iend - ioff
		//
		if in == 1 {
			//           1x1 block
			if (*wl) >= d.Get(ibegin-1)-(*pivmin) {
				nwl = nwl + 1
			}
			if (*wu) >= d.Get(ibegin-1)-(*pivmin) {
				nwu = nwu + 1
			}
			if irange == allrng || ((*wl) < d.Get(ibegin-1)-(*pivmin) && (*wu) >= d.Get(ibegin-1)-(*pivmin)) {
				(*m) = (*m) + 1
				w.Set((*m)-1, d.Get(ibegin-1))
				werr.Set((*m)-1, zero)
				//              The gap for a single block doesn't matter for the later
				//              algorithm and is assigned an arbitrary large value
				(*iblock)[(*m)-1] = jblk
				(*indexw)[(*m)-1] = 1
			}
			//        Disabled 2x2 case because of a failure on the following matrix
			//        RANGE = 'I', IL = IU = 4
			//          Original Tridiagonal, d = [
			//           -0.150102010615740E+00
			//           -0.849897989384260E+00
			//           -0.128208148052635E-15
			//            0.128257718286320E-15
			//          ];
			//          e = [
			//           -0.357171383266986E+00
			//           -0.180411241501588E-15
			//           -0.175152352710251E-15
			//          ];
			//
			//         ELSE IF( IN.EQ.2 ) THEN
			//*           2x2 block
			//            DISC = SQRT( (HALF*(D(IBEGIN)-D(IEND)))**2 + E(IBEGIN)**2 )
			//            TMP1 = HALF*(D(IBEGIN)+D(IEND))
			//            L1 = TMP1 - DISC
			//            IF( WL.GE. L1-PIVMIN )
			//     $         NWL = NWL + 1
			//            IF( WU.GE. L1-PIVMIN )
			//     $         NWU = NWU + 1
			//            IF( IRANGE.EQ.ALLRNG .OR. ( WL.LT.L1-PIVMIN .AND. WU.GE.
			//     $          L1-PIVMIN ) ) THEN
			//               M = M + 1
			//               W( M ) = L1
			//*              The uncertainty of eigenvalues of a 2x2 matrix is very small
			//               WERR( M ) = EPS * ABS( W( M ) ) * TWO
			//               IBLOCK( M ) = JBLK
			//               INDEXW( M ) = 1
			//            ENDIF
			//            L2 = TMP1 + DISC
			//            IF( WL.GE. L2-PIVMIN )
			//     $         NWL = NWL + 1
			//            IF( WU.GE. L2-PIVMIN )
			//     $         NWU = NWU + 1
			//            IF( IRANGE.EQ.ALLRNG .OR. ( WL.LT.L2-PIVMIN .AND. WU.GE.
			//     $          L2-PIVMIN ) ) THEN
			//               M = M + 1
			//               W( M ) = L2
			//*              The uncertainty of eigenvalues of a 2x2 matrix is very small
			//               WERR( M ) = EPS * ABS( W( M ) ) * TWO
			//               IBLOCK( M ) = JBLK
			//               INDEXW( M ) = 2
			//            ENDIF
		} else {
			//           General Case - block of size IN >= 2
			//           Compute local Gerschgorin interval and use it as the initial
			//           interval for DLAEBZ
			gu = d.Get(ibegin - 1)
			gl = d.Get(ibegin - 1)
			tmp1 = zero
			for j = ibegin; j <= iend; j++ {
				gl = minf64(gl, gers.Get(2*j-1-1))
				gu = maxf64(gu, gers.Get(2*j-1))
			}
			//           [JAN/28/2009]
			//           change SPDIAM by TNORM in lines 2 and 3 thereafter
			//           line 1: remove computation of SPDIAM (not useful anymore)
			//           SPDIAM = GU - GL
			//           GL = GL - FUDGE*SPDIAM*EPS*IN - FUDGE*PIVMIN
			//           GU = GU + FUDGE*SPDIAM*EPS*IN + FUDGE*PIVMIN
			gl = gl - fudge*tnorm*eps*float64(in) - fudge*(*pivmin)
			gu = gu + fudge*tnorm*eps*float64(in) + fudge*(*pivmin)
			//
			if irange > 1 {
				if gu < (*wl) {
					//                 the local block contains none of the wanted eigenvalues
					nwl = nwl + in
					nwu = nwu + in
					goto label70
				}
				//              refine search interval if possible, only _range (WL,WU] matters
				gl = maxf64(gl, *wl)
				gu = minf64(gu, *wu)
				if gl >= gu {
					goto label70
				}
			}
			//           Find negcount of initial interval boundaries GL and GU
			work.Set((*n)+1-1, gl)
			work.Set((*n)+in+1-1, gu)
			_iblockm1 := (*iblock)[(*m)+1-1:]
			Dlaebz(func() *int { y := 1; return &y }(), func() *int { y := 0; return &y }(), &in, &in, func() *int { y := 1; return &y }(), &nb, &atoli, &rtoli, pivmin, d.Off(ibegin-1), e.Off(ibegin-1), e2.Off(ibegin-1), &idumma, work.MatrixOff((*n)+1-1, in, opts), work.Off((*n)+2*in+1-1), &im, iwork, w.Off((*m)+1-1), &_iblockm1, &iinfo)
			if iinfo != 0 {
				(*info) = iinfo
				return
			}

			nwl = nwl + (*iwork)[0]
			nwu = nwu + (*iwork)[in+1-1]
			iwoff = (*m) - (*iwork)[0]
			//           Compute Eigenvalues
			itmax = int((math.Log(gu-gl+(*pivmin))-math.Log(*pivmin))/math.Log(two)) + 2
			_iblockm1 = (*iblock)[(*m)+1-1:]
			Dlaebz(func() *int { y := 2; return &y }(), &itmax, &in, &in, func() *int { y := 1; return &y }(), &nb, &atoli, &rtoli, pivmin, d.Off(ibegin-1), e.Off(ibegin-1), e2.Off(ibegin-1), &idumma, work.MatrixOff((*n)+1-1, in, opts), work.Off((*n)+2*in+1-1), &iout, iwork, w.Off((*m)+1-1), &_iblockm1, &iinfo)
			if iinfo != 0 {
				(*info) = iinfo
				return
			}

			//           Copy eigenvalues into W and IBLOCK
			//           Use -JBLK for block number for unconverged eigenvalues.
			//           Loop over the number of output intervals from DLAEBZ
			for j = 1; j <= iout; j++ {
				//              eigenvalue approximation is middle point of interval
				tmp1 = half * (work.Get(j+(*n)-1) + work.Get(j+in+(*n)-1))
				//              semi length of error interval
				tmp2 = half * math.Abs(work.Get(j+(*n)-1)-work.Get(j+in+(*n)-1))
				if j > iout-iinfo {
					//                 Flag non-convergence.
					ncnvrg = true
					ib = -jblk
				} else {
					ib = jblk
				}
				for je = (*iwork)[j-1] + 1 + iwoff; je <= (*iwork)[j+in-1]+iwoff; je++ {
					w.Set(je-1, tmp1)
					werr.Set(je-1, tmp2)
					(*indexw)[je-1] = je - iwoff
					(*iblock)[je-1] = ib
				}
			}

			(*m) = (*m) + im
		}
	label70:
	}
	//     If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
	//     If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
	if irange == indrng {
		idiscl = (*il) - 1 - nwl
		idiscu = nwu - (*iu)

		if idiscl > 0 {
			im = 0
			for je = 1; je <= (*m); je++ {
				//              Remove some of the smallest eigenvalues from the left so that
				//              at the end IDISCL =0. Move all eigenvalues up to the left.
				if w.Get(je-1) <= wlu && idiscl > 0 {
					idiscl = idiscl - 1
				} else {
					im = im + 1
					w.Set(im-1, w.Get(je-1))
					werr.Set(im-1, werr.Get(je-1))
					(*indexw)[im-1] = (*indexw)[je-1]
					(*iblock)[im-1] = (*iblock)[je-1]
				}
			}
			(*m) = im
		}
		if idiscu > 0 {
			//           Remove some of the largest eigenvalues from the right so that
			//           at the end IDISCU =0. Move all eigenvalues up to the left.
			im = (*m) + 1
			for je = (*m); je >= 1; je-- {
				if w.Get(je-1) >= wul && idiscu > 0 {
					idiscu = idiscu - 1
				} else {
					im = im - 1
					w.Set(im-1, w.Get(je-1))
					werr.Set(im-1, werr.Get(je-1))
					(*indexw)[im-1] = (*indexw)[je-1]
					(*iblock)[im-1] = (*iblock)[je-1]
				}
			}
			jee = 0
			for je = im; je <= (*m); je++ {
				jee = jee + 1
				w.Set(jee-1, w.Get(je-1))
				werr.Set(jee-1, werr.Get(je-1))
				(*indexw)[jee-1] = (*indexw)[je-1]
				(*iblock)[jee-1] = (*iblock)[je-1]
			}
			(*m) = (*m) - im + 1
		}
		if idiscl > 0 || idiscu > 0 {
			//           Code to deal with effects of bad arithmetic. (If N(w) is
			//           monotone non-decreasing, this should never happen.)
			//           Some low eigenvalues to be discarded are not in (WL,WLU],
			//           or high eigenvalues to be discarded are not in (WUL,WU]
			//           so just kill off the smallest IDISCL/largest IDISCU
			//           eigenvalues, by marking the corresponding IBLOCK = 0
			if idiscl > 0 {
				wkill = (*wu)
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
				wkill = (*wl)
				for jdisc = 1; jdisc <= idiscu; jdisc++ {
					iw = 0
					for je = 1; je <= (*m); je++ {
						if (*iblock)[je-1] != 0 && (w.Get(je-1) >= wkill || iw == 0) {
							iw = je
							wkill = w.Get(je - 1)
						}
					}
					(*iblock)[iw-1] = 0
				}
			}
			//           Now erase all eigenvalues with IBLOCK set to zero
			im = 0
			for je = 1; je <= (*m); je++ {
				if (*iblock)[je-1] != 0 {
					im = im + 1
					w.Set(im-1, w.Get(je-1))
					werr.Set(im-1, werr.Get(je-1))
					(*indexw)[im-1] = (*indexw)[je-1]
					(*iblock)[im-1] = (*iblock)[je-1]
				}
			}
			(*m) = im
		}
		if idiscl < 0 || idiscu < 0 {
			toofew = true
		}
	}

	if (irange == allrng && (*m) != (*n)) || (irange == indrng && (*m) != (*iu)-(*il)+1) {
		toofew = true
	}
	//     If ORDER='B', do nothing the eigenvalues are already sorted by
	//        block.
	//     If ORDER='E', sort the eigenvalues from smallest to largest
	if order == 'E' && (*nsplit) > 1 {
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
				tmp2 = werr.Get(ie - 1)
				itmp1 = (*iblock)[ie-1]
				itmp2 = (*indexw)[ie-1]
				w.Set(ie-1, w.Get(je-1))
				werr.Set(ie-1, werr.Get(je-1))
				(*iblock)[ie-1] = (*iblock)[je-1]
				(*indexw)[ie-1] = (*indexw)[je-1]
				w.Set(je-1, tmp1)
				werr.Set(je-1, tmp2)
				(*iblock)[je-1] = itmp1
				(*indexw)[je-1] = itmp2
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
