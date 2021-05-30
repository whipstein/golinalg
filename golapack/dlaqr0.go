package golapack

import (
	"golinalg/mat"
	"math"
)

// Dlaqr0 computes the eigenvalues of a Hessenberg matrix H
//    and, optionally, the matrices T and Z from the Schur decomposition
//    H = Z T Z**T, where T is an upper quasi-triangular matrix (the
//    Schur form), and Z is the orthogonal matrix of Schur vectors.
//
//    Optionally Z may be postmultiplied into an input orthogonal
//    matrix Q so that this routine can give the Schur factorization
//    of a matrix A which has been reduced to the Hessenberg form H
//    by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
func Dlaqr0(wantt, wantz bool, n, ilo, ihi *int, h *mat.Matrix, ldh *int, wr, wi *mat.Vector, iloz, ihiz *int, z *mat.Matrix, ldz *int, work *mat.Vector, lwork, info *int) {
	var sorted bool
	var aa, bb, cc, cs, dd, one, sn, ss, swap, wilk1, wilk2, zero float64
	var i, inf, it, itmax, k, kacc22, kbot, kdu, kexnw, kexsh, ks, kt, ktop, ku, kv, kwh, kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh, nho, nibble, nmin, ns, nsmax, nsr, ntiny, nve, nw, nwmax, nwr, nwupbd int
	jbcmpz := make([]byte, 2)

	zdum := mf(1, 1, opts)

	//     ==== Matrices of order NTINY or smaller must be processed by
	//     .    DLAHQR because of insufficient subdiagonal scratch space.
	//     .    (This is a hard limit.) ====
	ntiny = 11

	//     ==== Exceptional deflation windows:  try to cure rare
	//     .    slow convergence by varying the size of the
	//     .    deflation window after KEXNW iterations. ====
	kexnw = 5

	//     ==== Exceptional shifts: try to cure rare slow convergence
	//     .    with ad-hoc exceptional shifts every KEXSH iterations.
	//     .    ====
	kexsh = 6

	//     ==== The constants WILK1 and WILK2 are used to form the
	//     .    exceptional shifts. ====
	wilk1 = 0.75
	wilk2 = -0.4375
	zero = 0.0
	one = 1.0

	(*info) = 0

	//     ==== Quick return for N = 0: nothing to do. ====
	if (*n) == 0 {
		work.Set(0, one)
		return
	}

	if (*n) <= ntiny {
		//        ==== Tiny matrices must use DLAHQR. ====
		lwkopt = 1
		if (*lwork) != -1 {
			Dlahqr(wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info)
		}
	} else {
		//        ==== Use small bulge multi-shift QR with aggressive early
		//        .    deflation on larger-than-tiny matrices. ====
		//
		//        ==== Hope for the best. ====
		(*info) = 0

		//        ==== Set up job flags for ILAENV. ====
		if wantt {
			jbcmpz[0] = 'S'
		} else {
			jbcmpz[0] = 'E'
		}
		if wantz {
			jbcmpz[1] = 'V'
		} else {
			jbcmpz[1] = 'N'
		}

		//        ==== NWR = recommended deflation window size.  At this
		//        .    point,  N .GT. NTINY = 11, so there is enough
		//        .    subdiagonal workspace for NWR.GE.2 as required.
		//        .    (In fact, there is enough subdiagonal space for
		//        .    NWR.GE.3.) ====
		nwr = Ilaenv(func() *int { y := 13; return &y }(), []byte("DLAQR0"), jbcmpz, n, ilo, ihi, lwork)
		nwr = maxint(2, nwr)
		nwr = minint((*ihi)-(*ilo)+1, ((*n)-1)/3, nwr)

		//        ==== NSR = recommended number of simultaneous shifts.
		//        .    At this point N .GT. NTINY = 11, so there is at
		//        .    enough subdiagonal workspace for NSR to be even
		//        .    and greater than or equal to two as required. ====
		nsr = Ilaenv(func() *int { y := 15; return &y }(), []byte("DLAQR0"), jbcmpz, n, ilo, ihi, lwork)
		nsr = minint(nsr, ((*n)+6)/9, (*ihi)-(*ilo))
		nsr = maxint(2, nsr-(nsr%2))

		//        ==== Estimate optimal workspace ====
		//
		//        ==== Workspace query call to DLAQR3 ====
		Dlaqr3(wantt, wantz, n, ilo, ihi, toPtr(nwr+1), h, ldh, iloz, ihiz, z, ldz, &ls, &ld, wr, wi, h, ldh, n, h, ldh, n, h, ldh, work, toPtr(-1))

		//        ==== Optimal workspace = MAX(DLAQR5, DLAQR3) ====
		lwkopt = maxint(3*nsr/2, int(work.Get(0)))

		//        ==== Quick return in case of workspace query. ====
		if (*lwork) == -1 {
			work.Set(0, float64(lwkopt))
			return
		}

		//        ==== DLAHQR/DLAQR0 crossover point ====
		nmin = Ilaenv(func() *int { y := 12; return &y }(), []byte("DLAQR0"), jbcmpz, n, ilo, ihi, lwork)
		nmin = maxint(ntiny, nmin)

		//        ==== Nibble crossover point ====
		nibble = Ilaenv(func() *int { y := 14; return &y }(), []byte("DLAQR0"), jbcmpz, n, ilo, ihi, lwork)
		nibble = maxint(0, nibble)

		//        ==== Accumulate reflections during ttswp?  Use block
		//        .    2-by-2 structure during matrix-matrix multiply? ====
		kacc22 = Ilaenv(func() *int { y := 16; return &y }(), []byte("DLAQR0"), jbcmpz, n, ilo, ihi, lwork)
		kacc22 = maxint(0, kacc22)
		kacc22 = minint(2, kacc22)

		//        ==== NWMAX = the largest possible deflation window for
		//        .    which there is sufficient workspace. ====
		nwmax = minint(((*n)-1)/3, (*lwork)/2)
		nw = nwmax

		//        ==== NSMAX = the Largest number of simultaneous shifts
		//        .    for which there is sufficient workspace. ====
		nsmax = minint(((*n)+6)/9, 2*(*lwork)/3)
		nsmax = nsmax - (nsmax % 2)

		//        ==== NDFL: an iteration count restarted at deflation. ====
		ndfl = 1

		//        ==== ITMAX = iteration limit ====
		itmax = maxint(30, 2*kexsh) * maxint(10, (*ihi)-(*ilo)+1)

		//        ==== Last row and column in the active block ====
		kbot = (*ihi)

		//        ==== Main Loop ====
		for it = 1; it <= itmax; it++ {
			//           ==== Done when KBOT falls below ILO ====
			if kbot < (*ilo) {
				goto label90
			}

			//           ==== Locate active block ====
			for k = kbot; k >= (*ilo)+1; k-- {
				if h.Get(k-1, k-1-1) == zero {
					goto label20
				}
			}
			k = (*ilo)
		label20:
			;
			ktop = k

			//           ==== Select deflation window size:
			//           .    Typical Case:
			//           .      If possible and advisable, nibble the entire
			//           .      active block.  If not, use size MIN(NWR,NWMAX)
			//           .      or MIN(NWR+1,NWMAX) depending upon which has
			//           .      the smaller corresponding subdiagonal entry
			//           .      (a heuristic).
			//           .
			//           .    Exceptional Case:
			//           .      If there have been no deflations in KEXNW or
			//           .      more iterations, then vary the deflation window
			//           .      size.   At first, because, larger windows are,
			//           .      in general, more powerful than smaller ones,
			//           .      rapidly increase the window to the maximum possible.
			//           .      Then, gradually reduce the window size. ====
			nh = kbot - ktop + 1
			nwupbd = minint(nh, nwmax)
			if ndfl < kexnw {
				nw = minint(nwupbd, nwr)
			} else {
				nw = minint(nwupbd, 2*nw)
			}
			if nw < nwmax {
				if nw >= nh-1 {
					nw = nh
				} else {
					kwtop = kbot - nw + 1
					if math.Abs(h.Get(kwtop-1, kwtop-1-1)) > math.Abs(h.Get(kwtop-1-1, kwtop-2-1)) {
						nw = nw + 1
					}
				}
			}
			if ndfl < kexnw {
				ndec = -1
			} else if ndec >= 0 || nw >= nwupbd {
				ndec = ndec + 1
				if nw-ndec < 2 {
					ndec = 0
				}
				nw = nw - ndec
			}

			//           ==== Aggressive early deflation:
			//           .    split workspace under the subdiagonal into
			//           .      - an nw-by-nw work array V in the lower
			//           .        left-hand-corner,
			//           .      - an NW-by-at-least-NW-but-more-is-better
			//           .        (NW-by-NHO) horizontal work array along
			//           .        the bottom edge,
			//           .      - an at-least-NW-but-more-is-better (NHV-by-NW)
			//           .        vertical work array along the left-hand-edge.
			//           .        ====
			kv = (*n) - nw + 1
			kt = nw + 1
			nho = ((*n) - nw - 1) - kt + 1
			kwv = nw + 2
			nve = ((*n) - nw) - kwv + 1

			//           ==== Aggressive early deflation ====
			Dlaqr3(wantt, wantz, n, &ktop, &kbot, &nw, h, ldh, iloz, ihiz, z, ldz, &ls, &ld, wr, wi, h.Off(kv-1, 0), ldh, &nho, h.Off(kv-1, kt-1), ldh, &nve, h.Off(kwv-1, 0), ldh, work, lwork)

			//           ==== Adjust KBOT accounting for new deflations. ====
			kbot = kbot - ld

			//           ==== KS points to the shifts. ====
			ks = kbot - ls + 1

			//           ==== Skip an expensive QR sweep if there is a (partly
			//           .    heuristic) reason to expect that many eigenvalues
			//           .    will deflate without it.  Here, the QR sweep is
			//           .    skipped if many eigenvalues have just been deflated
			//           .    or if the remaining active block is small.
			if (ld == 0) || ((100*ld <= nw*nibble) && (kbot-ktop+1 > minint(nmin, nwmax))) {
				//              ==== NS = nominal number of simultaneous shifts.
				//              .    This may be lowered (slightly) if DLAQR3
				//              .    did not provide that many shifts. ====
				ns = minint(nsmax, nsr, maxint(2, kbot-ktop))
				ns = ns - (ns % 2)

				//              ==== If there have been no deflations
				//              .    in a multiple of KEXSH iterations,
				//              .    then try exceptional shifts.
				//              .    Otherwise use shifts provided by
				//              .    DLAQR3 above or from the eigenvalues
				//              .    of a trailing principal submatrix. ====
				if (ndfl % kexsh) == 0 {
					ks = kbot - ns + 1
					for i = kbot; i >= maxint(ks+1, ktop+2); i -= 2 {
						ss = math.Abs(h.Get(i-1, i-1-1)) + math.Abs(h.Get(i-1-1, i-2-1))
						aa = wilk1*ss + h.Get(i-1, i-1)
						bb = ss
						cc = wilk2 * ss
						dd = aa
						Dlanv2(&aa, &bb, &cc, &dd, wr.GetPtr(i-1-1), wi.GetPtr(i-1-1), wr.GetPtr(i-1), wi.GetPtr(i-1), &cs, &sn)
					}
					if ks == ktop {
						wr.Set(ks+1-1, h.Get(ks+1-1, ks+1-1))
						wi.Set(ks+1-1, zero)
						wr.Set(ks-1, wr.Get(ks+1-1))
						wi.Set(ks-1, wi.Get(ks+1-1))
					}
				} else {

					//                 ==== Got NS/2 or fewer shifts? Use DLAQR4 or
					//                 .    DLAHQR on a trailing principal submatrix to
					//                 .    get more. (Since NS.LE.NSMAX.LE.(N+6)/9,
					//                 .    there is enough space below the subdiagonal
					//                 .    to fit an NS-by-NS scratch array.) ====
					if kbot-ks+1 <= ns/2 {
						ks = kbot - ns + 1
						kt = (*n) - ns + 1
						Dlacpy('A', &ns, &ns, h.Off(ks-1, ks-1), ldh, h.Off(kt-1, 0), ldh)
						if ns > nmin {
							Dlaqr4(false, false, &ns, func() *int { y := 1; return &y }(), &ns, h.Off(kt-1, 0), ldh, wr.Off(ks-1), wi.Off(ks-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), zdum, func() *int { y := 1; return &y }(), work, lwork, &inf)
						} else {
							Dlahqr(false, false, &ns, func() *int { y := 1; return &y }(), &ns, h.Off(kt-1, 0), ldh, wr.Off(ks-1), wi.Off(ks-1), func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), zdum, func() *int { y := 1; return &y }(), &inf)
						}
						ks = ks + inf

						//                    ==== In case of a rare QR failure use
						//                    .    eigenvalues of the trailing 2-by-2
						//                    .    principal submatrix.  ====
						if ks >= kbot {
							aa = h.Get(kbot-1-1, kbot-1-1)
							cc = h.Get(kbot-1, kbot-1-1)
							bb = h.Get(kbot-1-1, kbot-1)
							dd = h.Get(kbot-1, kbot-1)
							Dlanv2(&aa, &bb, &cc, &dd, wr.GetPtr(kbot-1-1), wi.GetPtr(kbot-1-1), wr.GetPtr(kbot-1), wi.GetPtr(kbot-1), &cs, &sn)
							ks = kbot - 1
						}
					}

					if kbot-ks+1 > ns {
						//                    ==== Sort the shifts (Helps a little)
						//                    .    Bubble sort keeps complex conjugate
						//                    .    pairs together. ====
						sorted = false
						for k = kbot; k >= ks+1; k-- {
							if sorted {
								goto label60
							}
							sorted = true
							for i = ks; i <= k-1; i++ {
								if math.Abs(wr.Get(i-1))+math.Abs(wi.Get(i-1)) < math.Abs(wr.Get(i+1-1))+math.Abs(wi.Get(i+1-1)) {
									sorted = false

									swap = wr.Get(i - 1)
									wr.Set(i-1, wr.Get(i+1-1))
									wr.Set(i+1-1, swap)

									swap = wi.Get(i - 1)
									wi.Set(i-1, wi.Get(i+1-1))
									wi.Set(i+1-1, swap)
								}
							}
						}
					label60:
					}

					//                 ==== Shuffle shifts into pairs of real shifts
					//                 .    and pairs of complex conjugate shifts
					//                 .    assuming complex conjugate shifts are
					//                 .    already adjacent to one another. (Yes,
					//                 .    they are.)  ====
					for i = kbot; i >= ks+2; i -= 2 {
						if wi.Get(i-1) != -wi.Get(i-1-1) {

							swap = wr.Get(i - 1)
							wr.Set(i-1, wr.Get(i-1-1))
							wr.Set(i-1-1, wr.Get(i-2-1))
							wr.Set(i-2-1, swap)

							swap = wi.Get(i - 1)
							wi.Set(i-1, wi.Get(i-1-1))
							wi.Set(i-1-1, wi.Get(i-2-1))
							wi.Set(i-2-1, swap)
						}
					}
				}

				//              ==== If there are only two shifts and both are
				//              .    real, then use only one.  ====
				if kbot-ks+1 == 2 {
					if wi.Get(kbot-1) == zero {
						if math.Abs(wr.Get(kbot-1)-h.Get(kbot-1, kbot-1)) < math.Abs(wr.Get(kbot-1-1)-h.Get(kbot-1, kbot-1)) {
							wr.Set(kbot-1-1, wr.Get(kbot-1))
						} else {
							wr.Set(kbot-1, wr.Get(kbot-1-1))
						}
					}
				}

				//              ==== Use up to NS of the the smallest magnitude
				//              .    shifts.  If there aren't NS shifts available,
				//              .    then use them all, possibly dropping one to
				//              .    make the number of shifts even. ====
				ns = minint(ns, kbot-ks+1)
				ns = ns - (ns % 2)
				ks = kbot - ns + 1

				//              ==== Small-bulge multi-shift QR sweep:
				//              .    split workspace under the subdiagonal into
				//              .    - a KDU-by-KDU work array U in the lower
				//              .      left-hand-corner,
				//              .    - a KDU-by-at-least-KDU-but-more-is-better
				//              .      (KDU-by-NHo) horizontal work array WH along
				//              .      the bottom edge,
				//              .    - and an at-least-KDU-but-more-is-better-by-KDU
				//              .      (NVE-by-KDU) vertical work WV arrow along
				//              .      the left-hand-edge. ====
				kdu = 3*ns - 3
				ku = (*n) - kdu + 1
				kwh = kdu + 1
				nho = ((*n) - kdu + 1 - 4) - (kdu + 1) + 1
				kwv = kdu + 4
				nve = (*n) - kdu - kwv + 1

				//              ==== Small-bulge multi-shift QR sweep ====
				Dlaqr5(wantt, wantz, &kacc22, n, &ktop, &kbot, &ns, wr.Off(ks-1), wi.Off(ks-1), h, ldh, iloz, ihiz, z, ldz, work.MatrixOff(3, 3, opts), func() *int { y := 3; return &y }(), h.Off(ku-1, 0), ldh, &nve, h.Off(kwv-1, 0), ldh, &nho, h.Off(ku-1, kwh-1), ldh)
			}

			//           ==== Note progress (or the lack of it). ====
			if ld > 0 {
				ndfl = 1
			} else {
				ndfl = ndfl + 1
			}

			//           ==== End of main loop ====
		}

		//        ==== Iteration limit exceeded.  Set INFO to show where
		//        .    the problem occurred and exit. ====
		(*info) = kbot
	label90:
	}

	//     ==== Return the optimal value of LWORK. ====
	work.Set(0, float64(lwkopt))
}
