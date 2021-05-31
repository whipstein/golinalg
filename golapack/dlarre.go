package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlarre To find the desired eigenvalues of a given real symmetric
// tridiagonal matrix T, DLARRE sets any "small" off-diagonal
// elements to zero, and for each unreduced block T_i, it finds
// (a) a suitable shift at one end of the block's spectrum,
// (b) the base representation, T_i - sigma_i I = L_i D_i L_i^T, and
// (c) eigenvalues of each L_i D_i L_i^T.
// The representations and eigenvalues found are then used by
// DSTEMR to compute the eigenvectors of T.
// The accuracy varies depending on whether bisection is used to
// find a few eigenvalues or the dqds algorithm (subroutine DLASQ2) to
// conpute all and then discard any unwanted one.
// As an added benefit, DLARRE also outputs the n
// Gerschgorin intervals for the matrices L_i D_i L_i^T.
func Dlarre(_range byte, n *int, vl, vu *float64, il, iu *int, d, e, e2 *mat.Vector, rtol1, rtol2, spltol *float64, nsplit *int, isplit *[]int, m *int, w, werr, wgap *mat.Vector, iblock, indexw *[]int, gers *mat.Vector, pivmin *float64, work *mat.Vector, iwork *[]int, info *int) {
	var forceb, norep, usedqd bool
	var avgap, bsrtol, clwdth, dmax, dpivot, eabs, emax, eold, eps, fac, four, fourth, fudge, gl, gu, half, hndrd, isleft, isrght, maxgrowth, one, pert, rtl, rtol, s1, s2, safmin, sgndef, sigma, spdiam, tau, tmp, tmp1, two, zero float64
	var allrng, cnt, cnt1, cnt2, i, ibegin, idum, iend, iinfo, in, indl, indrng, indu, irange, j, jblk, maxtry, mb, mm, valrng, wbegin, wend int
	iseed := make([]int, 4)

	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0
	hndrd = 100.0
	pert = 8.0
	half = one / two
	fourth = one / four
	fac = half
	maxgrowth = 64.0
	fudge = 2.0
	maxtry = 6
	allrng = 1
	indrng = 2
	valrng = 3

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
	}
	(*m) = 0
	//     Get machine constants
	safmin = Dlamch(SafeMinimum)
	eps = Dlamch(Precision)
	//     Set parameters
	rtl = math.Sqrt(eps)
	bsrtol = math.Sqrt(eps)
	//     Treat case of 1x1 matrix for quick return
	if (*n) == 1 {
		if (irange == allrng) || ((irange == valrng) && (d.Get(0) > (*vl)) && (d.Get(0) <= (*vu))) || ((irange == indrng) && ((*il) == 1) && ((*iu) == 1)) {
			(*m) = 1
			w.Set(0, d.Get(0))
			//           The computation error of the eigenvalue is zero
			werr.Set(0, zero)
			wgap.Set(0, zero)
			(*iblock)[0] = 1
			(*indexw)[0] = 1
			gers.Set(0, d.Get(0))
			gers.Set(1, d.Get(0))
		}
		//        store the shift for the initial RRR, which is zero in this case
		e.Set(0, zero)
		return
	}
	//     General case: tridiagonal matrix of order > 1
	//
	//     Init WERR, WGAP. Compute Gerschgorin intervals and spectral diameter.
	//     Compute maximum off-diagonal entry and pivmin.
	gl = d.Get(0)
	gu = d.Get(0)
	eold = zero
	emax = zero
	e.Set((*n)-1, zero)
	for i = 1; i <= (*n); i++ {
		werr.Set(i-1, zero)
		wgap.Set(i-1, zero)
		eabs = math.Abs(e.Get(i - 1))
		if eabs >= emax {
			emax = eabs
		}
		tmp1 = eabs + eold
		gers.Set(2*i-1-1, d.Get(i-1)-tmp1)
		gl = minf64(gl, gers.Get(2*i-1-1))
		gers.Set(2*i-1, d.Get(i-1)+tmp1)
		gu = maxf64(gu, gers.Get(2*i-1))
		eold = eabs
	}
	//     The minimum pivot allowed in the Sturm sequence for T
	(*pivmin) = safmin * maxf64(one, math.Pow(emax, 2))
	//     Compute spectral diameter. The Gerschgorin bounds give an
	//     estimate that is wrong by at most a factor of SQRT(2)
	spdiam = gu - gl
	//     Compute splitting points
	Dlarra(n, d, e, e2, spltol, &spdiam, nsplit, isplit, &iinfo)
	//     Can force use of bisection instead of faster DQDS.
	//     Option left in the code for future multisection work.
	forceb = false
	//     Initialize USEDQD, DQDS should be used for ALLRNG unless someone
	//     explicitly wants bisection.
	usedqd = ((irange == allrng) && (!forceb))
	if (irange == allrng) && (!forceb) {
		//        Set interval [VL,VU] that contains all eigenvalues
		(*vl) = gl
		(*vu) = gu
	} else {
		//        We call DLARRD to find crude approximations to the eigenvalues
		//        in the desired _range. In case IRANGE = INDRNG, we also obtain the
		//        interval (VL,VU] that contains all the wanted eigenvalues.
		//        An interval [LEFT,RIGHT] has converged if
		//        RIGHT-LEFT.LT.RTOL*MAX(ABS(LEFT),ABS(RIGHT))
		//        DLARRD needs a WORK of size 4*N, IWORK of size 3*N
		Dlarrd(_range, 'B', n, vl, vu, il, iu, gers, &bsrtol, d, e, e2, pivmin, nsplit, isplit, &mm, w, werr, vl, vu, iblock, indexw, work, iwork, &iinfo)
		if iinfo != 0 {
			(*info) = -1
			return
		}
		//        Make sure that the entries M+1 to N in W, WERR, IBLOCK, INDEXW are 0
		for i = mm + 1; i <= (*n); i++ {
			w.Set(i-1, zero)
			werr.Set(i-1, zero)
			(*iblock)[i-1] = 0
			(*indexw)[i-1] = 0
		}
	}

	//     Loop over unreduced blocks
	ibegin = 1
	wbegin = 1
	for jblk = 1; jblk <= (*nsplit); jblk++ {
		iend = (*isplit)[jblk-1]
		in = iend - ibegin + 1
		//        1 X 1 block
		if in == 1 {
			if (irange == allrng) || ((irange == valrng) && (d.Get(ibegin-1) > (*vl)) && (d.Get(ibegin-1) <= (*vu))) || ((irange == indrng) && ((*iblock)[wbegin-1] == jblk)) {
				(*m) = (*m) + 1
				w.Set((*m)-1, d.Get(ibegin-1))
				werr.Set((*m)-1, zero)
				//              The gap for a single block doesn't matter for the later
				//              algorithm and is assigned an arbitrary large value
				wgap.Set((*m)-1, zero)
				(*iblock)[(*m)-1] = jblk
				(*indexw)[(*m)-1] = 1
				wbegin = wbegin + 1
			}
			//           E( IEND ) holds the shift for the initial RRR
			e.Set(iend-1, zero)
			ibegin = iend + 1
			goto label170
		}

		//        Blocks of size larger than 1x1
		//
		//        E( IEND ) will hold the shift for the initial RRR, for now set it =0
		e.Set(iend-1, zero)

		//        Find local outer bounds GL,GU for the block
		gl = d.Get(ibegin - 1)
		gu = d.Get(ibegin - 1)
		for i = ibegin; i <= iend; i++ {
			gl = minf64(gers.Get(2*i-1-1), gl)
			gu = maxf64(gers.Get(2*i-1), gu)
		}
		spdiam = gu - gl
		if !((irange == allrng) && (!forceb)) {
			//           Count the number of eigenvalues in the current block.
			mb = 0
			for i = wbegin; i <= mm; i++ {
				if (*iblock)[i-1] == jblk {
					mb = mb + 1
				} else {
					goto label21
				}
			}
		label21:
			;
			if mb == 0 {
				//              No eigenvalue in the current block lies in the desired _range
				//              E( IEND ) holds the shift for the initial RRR
				e.Set(iend-1, zero)
				ibegin = iend + 1
				goto label170
			} else {
				//              Decide whether dqds or bisection is more efficient
				usedqd = ((float64(mb) > fac*float64(in)) && (!forceb))
				wend = wbegin + mb - 1
				//              Calculate gaps for the current block
				//              In later stages, when representations for individual
				//              eigenvalues are different, we use SIGMA = E( IEND ).
				sigma = zero
				for i = wbegin; i <= wend-1; i++ {
					wgap.Set(i-1, maxf64(zero, w.Get(i+1-1)-werr.Get(i+1-1)-(w.Get(i-1)+werr.Get(i-1))))
				}
				wgap.Set(wend-1, maxf64(zero, (*vu)-sigma-(w.Get(wend-1)+werr.Get(wend-1))))
				//              Find local index of the first and last desired evalue.
				indl = (*indexw)[wbegin-1]
				indu = (*indexw)[wend-1]
			}
		}
		if ((irange == allrng) && (!forceb)) || usedqd {
			//           Case of DQDS
			//           Find approximations to the extremal eigenvalues of the block
			Dlarrk(&in, func() *int { y := 1; return &y }(), &gl, &gu, d.Off(ibegin-1), e2.Off(ibegin-1), pivmin, &rtl, &tmp, &tmp1, &iinfo)
			if iinfo != 0 {
				(*info) = -1
				return
			}
			isleft = maxf64(gl, tmp-tmp1-hndrd*eps*math.Abs(tmp-tmp1))
			Dlarrk(&in, &in, &gl, &gu, d.Off(ibegin-1), e2.Off(ibegin-1), pivmin, &rtl, &tmp, &tmp1, &iinfo)
			if iinfo != 0 {
				(*info) = -1
				return
			}
			isrght = minf64(gu, tmp+tmp1+hndrd*eps*math.Abs(tmp+tmp1))
			//           Improve the estimate of the spectral diameter
			spdiam = isrght - isleft
		} else {
			//           Case of bisection
			//           Find approximations to the wanted extremal eigenvalues
			isleft = maxf64(gl, w.Get(wbegin-1)-werr.Get(wbegin-1)-hndrd*eps*math.Abs(w.Get(wbegin-1)-werr.Get(wbegin-1)))
			isrght = minf64(gu, w.Get(wend-1)+werr.Get(wend-1)+hndrd*eps*math.Abs(w.Get(wend-1)+werr.Get(wend-1)))
		}
		//        Decide whether the base representation for the current block
		//        L_JBLK D_JBLK L_JBLK^T = T_JBLK - sigma_JBLK I
		//        should be on the left or the right end of the current block.
		//        The strategy is to shift to the end which is "more populated"
		//        Furthermore, decide whether to use DQDS for the computation of
		//        the eigenvalue approximations at the end of DLARRE or bisection.
		//        dqds is chosen if all eigenvalues are desired or the number of
		//        eigenvalues to be computed is large compared to the blocksize.
		if (irange == allrng) && (!forceb) {
			//           If all the eigenvalues have to be computed, we use dqd
			usedqd = true
			//           INDL is the local index of the first eigenvalue to compute
			indl = 1
			indu = in
			//           MB =  number of eigenvalues to compute
			mb = in
			wend = wbegin + mb - 1
			//           Define 1/4 and 3/4 points of the spectrum
			s1 = isleft + fourth*spdiam
			s2 = isrght - fourth*spdiam
		} else {
			//           DLARRD has computed IBLOCK and INDEXW for each eigenvalue
			//           approximation.
			//           choose sigma
			if usedqd {
				s1 = isleft + fourth*spdiam
				s2 = isrght - fourth*spdiam
			} else {
				tmp = minf64(isrght, *vu) - maxf64(isleft, *vl)
				s1 = maxf64(isleft, *vl) + fourth*tmp
				s2 = minf64(isrght, *vu) - fourth*tmp
			}
		}
		//        Compute the negcount at the 1/4 and 3/4 points
		if mb > 1 {
			Dlarrc('T', &in, &s1, &s2, d.Off(ibegin-1), e.Off(ibegin-1), pivmin, &cnt, &cnt1, &cnt2, &iinfo)
		}
		if mb == 1 {
			sigma = gl
			sgndef = one
		} else if cnt1-indl >= indu-cnt2 {
			if (irange == allrng) && (!forceb) {
				sigma = maxf64(isleft, gl)
			} else if usedqd {
				//              use Gerschgorin bound as shift to get pos def matrix
				//              for dqds
				sigma = isleft
			} else {
				//              use approximation of the first desired eigenvalue of the
				//              block as shift
				sigma = maxf64(isleft, *vl)
			}
			sgndef = one
		} else {
			if (irange == allrng) && (!forceb) {
				sigma = minf64(isrght, gu)
			} else if usedqd {
				//              use Gerschgorin bound as shift to get neg def matrix
				//              for dqds
				sigma = isrght
			} else {
				//              use approximation of the first desired eigenvalue of the
				//              block as shift
				sigma = minf64(isrght, *vu)
			}
			sgndef = -one
		}
		//        An initial SIGMA has been chosen that will be used for computing
		//        T - SIGMA I = L D L^T
		//        Define the increment TAU of the shift in case the initial shift
		//        needs to be refined to obtain a factorization with not too much
		//        element growth.
		if usedqd {
			//           The initial SIGMA was to the outer end of the spectrum
			//           the matrix is definite and we need not retreat.
			tau = spdiam*eps*float64(*n) + two*(*pivmin)
			tau = maxf64(tau, two*eps*math.Abs(sigma))
		} else {
			if mb > 1 {
				clwdth = w.Get(wend-1) + werr.Get(wend-1) - w.Get(wbegin-1) - werr.Get(wbegin-1)
				avgap = math.Abs(clwdth / float64(wend-wbegin))
				if sgndef == one {
					tau = half * maxf64(wgap.Get(wbegin-1), avgap)
					tau = maxf64(tau, werr.Get(wbegin-1))
				} else {
					tau = half * maxf64(wgap.Get(wend-1-1), avgap)
					tau = maxf64(tau, werr.Get(wend-1))
				}
			} else {
				tau = werr.Get(wbegin - 1)
			}
		}

		for idum = 1; idum <= maxtry; idum++ {
			//           Compute L D L^T factorization of tridiagonal matrix T - sigma I.
			//           Store D in WORK(1:IN), L in WORK(IN+1:2*IN), and reciprocals of
			//           pivots in WORK(2*IN+1:3*IN)
			dpivot = d.Get(ibegin-1) - sigma
			work.Set(0, dpivot)
			dmax = math.Abs(work.Get(0))
			j = ibegin
			for i = 1; i <= in-1; i++ {
				work.Set(2*in+i-1, one/work.Get(i-1))
				tmp = e.Get(j-1) * work.Get(2*in+i-1)
				work.Set(in+i-1, tmp)
				dpivot = (d.Get(j+1-1) - sigma) - tmp*e.Get(j-1)
				work.Set(i+1-1, dpivot)
				dmax = maxf64(dmax, math.Abs(dpivot))
				j = j + 1
			}
			//           check for element growth
			if dmax > maxgrowth*spdiam {
				norep = true
			} else {
				norep = false
			}
			if usedqd && !norep {
				//              Ensure the definiteness of the representation
				//              All entries of D (of L D L^T) must have the same sign
				for i = 1; i <= in; i++ {
					tmp = sgndef * work.Get(i-1)
					if tmp < zero {
						norep = true
					}
				}
			}
			if norep {
				//              Note that in the case of IRANGE=ALLRNG, we use the Gerschgorin
				//              shift which makes the matrix definite. So we should end up
				//              here really only in the case of IRANGE = VALRNG or INDRNG.
				if idum == maxtry-1 {
					if sgndef == one {
						//                    The fudged Gerschgorin shift should succeed
						sigma = gl - fudge*spdiam*eps*float64(*n) - fudge*two*(*pivmin)
					} else {
						sigma = gu + fudge*spdiam*eps*float64(*n) + fudge*two*(*pivmin)
					}
				} else {
					sigma = sigma - sgndef*tau
					tau = two * tau
				}
			} else {
				//              an initial RRR is found
				goto label83
			}
		}
		//        if the program reaches this point, no base representation could be
		//        found in MAXTRY iterations.
		(*info) = 2
		return
	label83:
		;
		//        At this point, we have found an initial base representation
		//        T - SIGMA I = L D L^T with not too much element growth.
		//        Store the shift.
		e.Set(iend-1, sigma)
		//        Store D and L.
		goblas.Dcopy(&in, work, toPtr(1), d.Off(ibegin-1), toPtr(1))
		goblas.Dcopy(toPtr(in-1), work.Off(in+1-1), toPtr(1), e.Off(ibegin-1), toPtr(1))
		if mb > 1 {
			//           Perturb each entry of the base representation by a small
			//           (but random) relative amount to overcome difficulties with
			//           glued matrices.
			for i = 1; i <= 4; i++ {
				iseed[i-1] = 1
			}
			Dlarnv(func() *int { y := 2; return &y }(), &iseed, toPtr(2*in-1), work)
			for i = 1; i <= in-1; i++ {
				d.Set(ibegin+i-1-1, d.Get(ibegin+i-1-1)*(one+eps*pert*work.Get(i-1)))
				e.Set(ibegin+i-1-1, e.Get(ibegin+i-1-1)*(one+eps*pert*work.Get(in+i-1)))
			}
			d.Set(iend-1, d.Get(iend-1)*(one+eps*four*work.Get(in-1)))

		}

		//        Don't update the Gerschgorin intervals because keeping track
		//        of the updates would be too much work in DLARRV.
		//        We update W instead and use it to locate the proper Gerschgorin
		//        intervals.
		//        Compute the required eigenvalues of L D L' by bisection or dqds
		if !usedqd {
			//           If DLARRD has been used, shift the eigenvalue approximations
			//           according to their representation. This is necessary for
			//           a uniform DLARRV since dqds computes eigenvalues of the
			//           shifted representation. In DLARRV, W will always hold the
			//           UNshifted eigenvalue approximation.
			for j = wbegin; j <= wend; j++ {
				w.Set(j-1, w.Get(j-1)-sigma)
				werr.Set(j-1, werr.Get(j-1)+math.Abs(w.Get(j-1))*eps)
			}
			//           call DLARRB to reduce eigenvalue error of the approximations
			//           from DLARRD
			for i = ibegin; i <= iend-1; i++ {
				work.Set(i-1, d.Get(i-1)*math.Pow(e.Get(i-1), 2))
			}
			//           use bisection to find EV from INDL to INDU
			Dlarrb(&in, d.Off(ibegin-1), work.Off(ibegin-1), &indl, &indu, rtol1, rtol2, toPtr(indl-1), w.Off(wbegin-1), wgap.Off(wbegin-1), werr.Off(wbegin-1), work.Off(2*(*n)+1-1), iwork, pivmin, &spdiam, &in, &iinfo)
			if iinfo != 0 {
				(*info) = -4
				return
			}
			//           DLARRB computes all gaps correctly except for the last one
			//           Record distance to VU/GU
			wgap.Set(wend-1, maxf64(zero, ((*vu)-sigma)-(w.Get(wend-1)+werr.Get(wend-1))))
			for i = indl; i <= indu; i++ {
				(*m) = (*m) + 1
				(*iblock)[(*m)-1] = jblk
				(*indexw)[(*m)-1] = i
			}
		} else {
			//           Call dqds to get all eigs (and then possibly delete unwanted
			//           eigenvalues).
			//           Note that dqds finds the eigenvalues of the L D L^T representation
			//           of T to high relative accuracy. High relative accuracy
			//           might be lost when the shift of the RRR is subtracted to obtain
			//           the eigenvalues of T. However, T is not guaranteed to define its
			//           eigenvalues to high relative accuracy anyway.
			//           Set RTOL to the order of the tolerance used in DLASQ2
			//           This is an ESTIMATED error, the worst case bound is 4*N*EPS
			//           which is usually too large and requires unnecessary work to be
			//           done by bisection when computing the eigenvectors
			rtol = math.Log(float64(in)) * four * eps
			j = ibegin
			for i = 1; i <= in-1; i++ {
				work.Set(2*i-1-1, math.Abs(d.Get(j-1)))
				work.Set(2*i-1, e.Get(j-1)*e.Get(j-1)*work.Get(2*i-1-1))
				j = j + 1
			}
			work.Set(2*in-1-1, math.Abs(d.Get(iend-1)))
			work.Set(2*in-1, zero)
			Dlasq2(&in, work, &iinfo)
			if iinfo != 0 {
				//              If IINFO = -5 then an index is part of a tight cluster
				//              and should be changed. The index is in IWORK(1) and the
				//              gap is in WORK(N+1)
				(*info) = -5
				return
			} else {
				//              Test that all eigenvalues are positive as expected
				for i = 1; i <= in; i++ {
					if work.Get(i-1) < zero {
						(*info) = -6
						return
					}
				}
			}
			if sgndef > zero {
				for i = indl; i <= indu; i++ {
					(*m) = (*m) + 1
					w.Set((*m)-1, work.Get(in-i+1-1))
					(*iblock)[(*m)-1] = jblk
					(*indexw)[(*m)-1] = i
				}
			} else {
				for i = indl; i <= indu; i++ {
					(*m) = (*m) + 1
					w.Set((*m)-1, -work.Get(i-1))
					(*iblock)[(*m)-1] = jblk
					(*indexw)[(*m)-1] = i
				}
			}
			for i = (*m) - mb + 1; i <= (*m); i++ {
				//              the value of RTOL below should be the tolerance in DLASQ2
				werr.Set(i-1, rtol*math.Abs(w.Get(i-1)))
			}
			for i = (*m) - mb + 1; i <= (*m)-1; i++ {
				//              compute the right gap between the intervals
				wgap.Set(i-1, maxf64(zero, w.Get(i+1-1)-werr.Get(i+1-1)-(w.Get(i-1)+werr.Get(i-1))))
			}
			wgap.Set((*m)-1, maxf64(zero, ((*vu)-sigma)-(w.Get((*m)-1)+werr.Get((*m)-1))))
		}
		//        proceed with next block
		ibegin = iend + 1
		wbegin = wend + 1
	label170:
	}
}
