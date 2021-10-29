package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/mat"
)

// Zlaqr4 implements one level of recursion for ZLAQR0.
//    It is a complete implementation of the small bulge multi-shift
//    QR algorithm.  It may be called by ZLAQR0 and, for large enough
//    deflation window size, it may be called by ZLAQR3.  This
//    subroutine is identical to ZLAQR0 except that it calls ZLAQR2
//    instead of ZLAQR3.
//
//    Zlaqr4 computes the eigenvalues of a Hessenberg matrix H
//    and, optionally, the matrices T and Z from the Schur decomposition
//    H = Z T Z**H, where T is an upper triangular matrix (the
//    Schur form), and Z is the unitary matrix of Schur vectors.
//
//    Optionally Z may be postmultiplied into an input unitary
//    matrix Q so that this routine can give the Schur factorization
//    of a matrix A which has been reduced to the Hessenberg form H
//    by the unitary matrix Q:  A = Q*H*Q**H = (QZ)*H*(QZ)**H.
func Zlaqr4(wantt, wantz bool, n, ilo, ihi int, h *mat.CMatrix, w *mat.CVector, iloz, ihiz int, z *mat.CMatrix, work *mat.CVector, lwork int) (info int) {
	var sorted bool
	var aa, bb, cc, dd, det, one, rtdisc, swap, tr2, zero complex128
	var s, two, wilk1 float64
	var i, inf, it, itmax, k, kacc22, kbot, kdu, kexnw, kexsh, ks, kt, ktop, ku, kv, kwh, kwtop, kwv, ld, ls, lwkopt, ndec, ndfl, nh, nho, nibble, nmin, ns, nsmax, nsr, ntiny, nve, nw, nwmax, nwr, nwupbd int

	jbcmpz := make([]byte, 2)
	zdum := cmf(1, 1, opts)

	//     ==== Matrices of order NTINY or smaller must be processed by
	//     .    ZLAHQR because of insufficient subdiagonal scratch space.
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

	//     ==== The constant WILK1 is used to form the exceptional
	//     .    shifts. ====
	wilk1 = 0.75
	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	two = 2.0

	//     ==== Quick return for N = 0: nothing to do. ====
	if n == 0 {
		work.Set(0, one)
		return
	}

	if n <= ntiny {
		//        ==== Tiny matrices must use ZLAHQR. ====
		lwkopt = 1
		if lwork != -1 {
			info = Zlahqr(wantt, wantz, n, ilo, ihi, h, w, iloz, ihiz, z)
		}
	} else {
		//        ==== Use small bulge multi-shift QR with aggressive early
		//        .    deflation on larger-than-tiny matrices. ====
		//
		//        ==== Hope for the best. ====

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
		nwr = Ilaenv(13, "Zlaqr4", jbcmpz, n, ilo, ihi, lwork)
		nwr = max(2, nwr)
		nwr = min(ihi-ilo+1, (n-1)/3, nwr)

		//        ==== NSR = recommended number of simultaneous shifts.
		//        .    At this point N .GT. NTINY = 11, so there is at
		//        .    enough subdiagonal workspace for NSR to be even
		//        .    and greater than or equal to two as required. ====
		nsr = Ilaenv(15, "Zlaqr4", jbcmpz, n, ilo, ihi, lwork)
		nsr = min(nsr, (n+6)/9, ihi-ilo)
		nsr = max(2, nsr-(nsr%2))

		//        ==== Estimate optimal workspace ====
		//
		//        ==== Workspace query call to ZLAQR2 ====
		ls, ld = Zlaqr2(wantt, wantz, n, ilo, ihi, nwr+1, h, iloz, ihiz, z, w, h, n, h, n, h, work, -1)

		//        ==== Optimal workspace = max(ZLAQR5, ZLAQR2) ====
		lwkopt = max(3*nsr/2, int(work.GetRe(0)))

		//        ==== Quick return in case of workspace query. ====
		if lwork == -1 {
			work.Set(0, complex(float64(lwkopt), 0))
			return
		}

		//        ==== ZLAHQR/ZLAQR0 crossover point ====
		nmin = Ilaenv(12, "Zlaqr4", jbcmpz, n, ilo, ihi, lwork)
		nmin = max(ntiny, nmin)

		//        ==== Nibble crossover point ====
		nibble = Ilaenv(14, "Zlaqr4", jbcmpz, n, ilo, ihi, lwork)
		nibble = max(0, nibble)

		//        ==== Accumulate reflections during ttswp?  Use block
		//        .    2-by-2 structure during matrix-matrix multiply? ====
		kacc22 = Ilaenv(16, "Zlaqr4", jbcmpz, n, ilo, ihi, lwork)
		kacc22 = max(0, kacc22)
		kacc22 = min(2, kacc22)

		//        ==== NWMAX = the largest possible deflation window for
		//        .    which there is sufficient workspace. ====
		nwmax = min((n-1)/3, lwork/2)
		nw = nwmax

		//        ==== NSMAX = the Largest number of simultaneous shifts
		//        .    for which there is sufficient workspace. ====
		nsmax = min((n+6)/9, 2*lwork/3)
		nsmax = nsmax - (nsmax % 2)

		//        ==== NDFL: an iteration count restarted at deflation. ====
		ndfl = 1

		//        ==== ITMAX = iteration limit ====
		itmax = max(30, 2*kexsh) * max(10, ihi-ilo+1)

		//        ==== Last row and column in the active block ====
		kbot = ihi

		//        ==== Main Loop ====
		for it = 1; it <= itmax; it++ {
			//           ==== Done when KBOT falls below ILO ====
			if kbot < ilo {
				goto label80
			}

			//           ==== Locate active block ====
			for k = kbot; k >= ilo+1; k-- {
				if h.Get(k-1, k-1-1) == zero {
					goto label20
				}
			}
			k = ilo
		label20:
			;
			ktop = k

			//           ==== Select deflation window size:
			//           .    Typical Case:
			//           .      If possible and advisable, nibble the entire
			//           .      active block.  If not, use size min(NWR,NWMAX)
			//           .      or min(NWR+1,NWMAX) depending upon which has
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
			nwupbd = min(nh, nwmax)
			if ndfl < kexnw {
				nw = min(nwupbd, nwr)
			} else {
				nw = min(nwupbd, 2*nw)
			}
			if nw < nwmax {
				if nw >= nh-1 {
					nw = nh
				} else {
					kwtop = kbot - nw + 1
					if cabs1(h.Get(kwtop-1, kwtop-1-1)) > cabs1(h.Get(kwtop-1-1, kwtop-2-1)) {
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
			kv = n - nw + 1
			kt = nw + 1
			nho = (n - nw - 1) - kt + 1
			kwv = nw + 2
			nve = (n - nw) - kwv + 1

			//           ==== Aggressive early deflation ====
			ls, ld = Zlaqr2(wantt, wantz, n, ktop, kbot, nw, h, iloz, ihiz, z, w, h.Off(kv-1, 0), nho, h.Off(kv-1, kt-1), nve, h.Off(kwv-1, 0), work, lwork)

			//           ==== Adjust KBOT accounting for new deflations. ====
			kbot = kbot - ld

			//           ==== KS points to the shifts. ====
			ks = kbot - ls + 1

			//           ==== Skip an expensive QR sweep if there is a (partly
			//           .    heuristic) reason to expect that many eigenvalues
			//           .    will deflate without it.  Here, the QR sweep is
			//           .    skipped if many eigenvalues have just been deflated
			//           .    or if the remaining active block is small.
			if (ld == 0) || ((100*ld <= nw*nibble) && (kbot-ktop+1 > min(nmin, nwmax))) {
				//              ==== NS = nominal number of simultaneous shifts.
				//              .    This may be lowered (slightly) if ZLAQR2
				//              .    did not provide that many shifts. ====
				ns = min(nsmax, nsr, max(2, kbot-ktop))
				ns = ns - (ns % 2)

				//              ==== If there have been no deflations
				//              .    in a multiple of KEXSH iterations,
				//              .    then try exceptional shifts.
				//              .    Otherwise use shifts provided by
				//              .    ZLAQR2 above or from the eigenvalues
				//              .    of a trailing principal submatrix. ====
				if (ndfl % kexsh) == 0 {
					ks = kbot - ns + 1
					for i = kbot; i >= ks+1; i -= 2 {
						w.Set(i-1, h.Get(i-1, i-1)+complex(wilk1*cabs1(h.Get(i-1, i-1-1)), 0))
						w.Set(i-1-1, w.Get(i-1))
					}
				} else {
					//                 ==== Got NS/2 or fewer shifts? Use ZLAHQR
					//                 .    on a trailing principal submatrix to
					//                 .    get more. (Since NS.LE.NSMAX.LE.(N+6)/9,
					//                 .    there is enough space below the subdiagonal
					//                 .    to fit an NS-by-NS scratch array.) ====
					if kbot-ks+1 <= ns/2 {
						ks = kbot - ns + 1
						kt = n - ns + 1
						Zlacpy(Full, ns, ns, h.Off(ks-1, ks-1), h.Off(kt-1, 0))
						inf = Zlahqr(false, false, ns, 1, ns, h.Off(kt-1, 0), w.Off(ks-1), 1, 1, zdum)
						ks = ks + inf

						//                    ==== In case of a rare QR failure use
						//                    .    eigenvalues of the trailing 2-by-2
						//                    .    principal submatrix.  Scale to avoid
						//                    .    overflows, underflows and subnormals.
						//                    .    (The scale factor S can not be zero,
						//                    .    because H(KBOT,KBOT-1) is nonzero.) ====
						if ks >= kbot {
							s = cabs1(h.Get(kbot-1-1, kbot-1-1)) + cabs1(h.Get(kbot-1, kbot-1-1)) + cabs1(h.Get(kbot-1-1, kbot-1)) + cabs1(h.Get(kbot-1, kbot-1))
							aa = h.Get(kbot-1-1, kbot-1-1) / complex(s, 0)
							cc = h.Get(kbot-1, kbot-1-1) / complex(s, 0)
							bb = h.Get(kbot-1-1, kbot-1) / complex(s, 0)
							dd = h.Get(kbot-1, kbot-1) / complex(s, 0)
							tr2 = (aa + dd) / complex(two, 0)
							det = (aa-tr2)*(dd-tr2) - bb*cc
							rtdisc = cmplx.Sqrt(-det)
							w.Set(kbot-1-1, (tr2+rtdisc)*complex(s, 0))
							w.Set(kbot-1, (tr2-rtdisc)*complex(s, 0))

							ks = kbot - 1
						}
					}

					if kbot-ks+1 > ns {
						//                    ==== Sort the shifts (Helps a little) ====
						sorted = false
						for k = kbot; k >= ks+1; k-- {
							if sorted {
								goto label60
							}
							sorted = true
							for i = ks; i <= k-1; i++ {
								if cabs1(w.Get(i-1)) < cabs1(w.Get(i)) {
									sorted = false
									swap = w.Get(i - 1)
									w.Set(i-1, w.Get(i))
									w.Set(i, swap)
								}
							}
						}
					label60:
					}
				}

				//              ==== If there are only two shifts, then use
				//              .    only one.  ====
				if kbot-ks+1 == 2 {
					if cabs1(w.Get(kbot-1)-h.Get(kbot-1, kbot-1)) < cabs1(w.Get(kbot-1-1)-h.Get(kbot-1, kbot-1)) {
						w.Set(kbot-1-1, w.Get(kbot-1))
					} else {
						w.Set(kbot-1, w.Get(kbot-1-1))
					}
				}

				//              ==== Use up to NS of the the smallest magnitude
				//              .    shifts.  If there aren't NS shifts available,
				//              .    then use them all, possibly dropping one to
				//              .    make the number of shifts even. ====
				ns = min(ns, kbot-ks+1)
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
				ku = n - kdu + 1
				kwh = kdu + 1
				nho = (n - kdu + 1 - 4) - (kdu + 1) + 1
				kwv = kdu + 4
				nve = n - kdu - kwv + 1

				//              ==== Small-bulge multi-shift QR sweep ====
				Zlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns, w.Off(ks-1), h, iloz, ihiz, z, work.CMatrix(3, opts), h.Off(ku-1, 0), nve, h.Off(kwv-1, 0), nho, h.Off(ku-1, kwh-1))
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
		info = kbot
	label80:
	}

	//     ==== Return the optimal value of LWORK. ====
	work.Set(0, complex(float64(lwkopt), 0))

	return
}
