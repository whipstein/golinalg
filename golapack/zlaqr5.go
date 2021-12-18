package golapack

import (
	"math"

	"github.com/whipstein/golinalg/mat"
)

// Zlaqr5 called by ZLAQR0, performs a
//    single small-bulge multi-shift QR sweep.
func Zlaqr5(wantt, wantz bool, kacc22, n, ktop, kbot, nshfts int, s *mat.CVector, h *mat.CMatrix, iloz, ihiz int, z, v, u *mat.CMatrix, nv int, wv *mat.CMatrix, nh int, wh *mat.CMatrix) {
	var accum, blk22, bmp22 bool
	var alpha, beta, one, refsum, zero complex128
	var h11, h12, h21, h22, rone, rzero, safmax, safmin, scl, smlnum, tst1, tst2, ulp float64
	var i2, i4, incol, j, j2, j4, jbot, jcol, jlen, jrow, jtop, k, k1, kdu, kms, knz, krcol, kzs, m, m22, mbot, mend, mstart, mtop, nbmps, ndcol, ns, nu int
	var err error

	vt := cvf(3)

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0

	//     ==== If there are no shifts, then there is nothing to do. ====
	if nshfts < 2 {
		return
	}

	//     ==== If the active block is empty or 1-by-1, then there
	//     .    is nothing to do. ====
	if ktop >= kbot {
		return
	}

	//     ==== NSHFTS is supposed to be even, but if it is odd,
	//     .    then simply reduce it by one.  ====
	ns = nshfts - (nshfts % 2)

	//     ==== Machine constants for deflation ====
	safmin = Dlamch(SafeMinimum)
	safmax = rone / safmin
	safmin, safmax = Dlabad(safmin, safmax)
	ulp = Dlamch(Precision)
	smlnum = safmin * (float64(n) / ulp)

	//     ==== Use accumulated reflections to update far-from-diagonal
	//     .    entries ? ====
	accum = (kacc22 == 1) || (kacc22 == 2)

	//     ==== If so, exploit the 2-by-2 block structure? ====
	blk22 = (ns > 2) && (kacc22 == 2)

	//     ==== clear trash ====
	if ktop+2 <= kbot {
		h.Set(ktop+2-1, ktop-1, zero)
	}

	//     ==== NBMPS = number of 2-shift bulges in the chain ====
	nbmps = ns / 2

	//     ==== KDU = width of slab ====
	kdu = 6*nbmps - 3

	//     ==== Create and chase chains of NBMPS bulges ====
	for incol = 3*(1-nbmps) + ktop - 1; incol <= kbot-2; incol += 3*nbmps - 2 {
		ndcol = incol + kdu
		if accum {
			Zlaset(Full, kdu, kdu, zero, one, u)
		}

		//        ==== Near-the-diagonal bulge chase.  The following loop
		//        .    performs the near-the-diagonal part of a small bulge
		//        .    multi-shift QR sweep.  Each 6*NBMPS-2 column diagonal
		//        .    chunk extends from column INCOL to column NDCOL
		//        .    (including both column INCOL and column NDCOL). The
		//        .    following loop chases a 3*NBMPS column long chain of
		//        .    NBMPS bulges 3*NBMPS-2 columns to the right.  (INCOL
		//        .    may be less than KTOP and and NDCOL may be greater than
		//        .    KBOT indicating phantom columns from which to chase
		//        .    bulges before they are actually introduced or to which
		//        .    to chase bulges beyond column KBOT.)  ====
		for krcol = incol; krcol <= min(incol+3*nbmps-3, kbot-2); krcol++ {
			//           ==== Bulges number MTOP to MBOT are active double implicit
			//           .    shift bulges.  There may or may not also be small
			//           .    2-by-2 bulge, if there is room.  The inactive bulges
			//           .    (if any) must wait until the active bulges have moved
			//           .    down the diagonal to make room.  The phantom matrix
			//           .    paradigm described above helps keep track.  ====
			mtop = max(1, ((ktop-1)-krcol+2)/3+1)
			mbot = min(nbmps, (kbot-krcol)/3)
			m22 = mbot + 1
			bmp22 = (mbot < nbmps) && (krcol+3*(m22-1)) == (kbot-2)

			//           ==== Generate reflections to chase the chain right
			//           .    one column.  (The minimum value of K is KTOP-1.) ====
			for m = mtop; m <= mbot; m++ {
				k = krcol + 3*(m-1)
				if k == ktop-1 {
					Zlaqr1(3, h.Off(ktop-1, ktop-1), s.Get(2*m-1-1), s.Get(2*m-1), v.Off(0, m-1).CVector())
					alpha = v.Get(0, m-1)
					alpha, *v.GetPtr(0, m-1) = Zlarfg(3, alpha, v.Off(1, m-1).CVector(), 1)
				} else {
					beta = h.Get(k, k-1)
					v.Set(1, m-1, h.Get(k+2-1, k-1))
					v.Set(2, m-1, h.Get(k+3-1, k-1))
					beta, *v.GetPtr(0, m-1) = Zlarfg(3, beta, v.Off(1, m-1).CVector(), 1)

					//                 ==== A Bulge may collapse because of vigilant
					//                 .    deflation or destructive underflow.  In the
					//                 .    underflow case, try the two-small-subdiagonals
					//                 .    trick to try to reinflate the bulge.  ====
					if h.Get(k+3-1, k-1) != zero || h.Get(k+3-1, k) != zero || h.Get(k+3-1, k+2-1) == zero {
						//                    ==== Typical case: not collapsed (yet). ====
						h.Set(k, k-1, beta)
						h.Set(k+2-1, k-1, zero)
						h.Set(k+3-1, k-1, zero)
					} else {
						//                    ==== Atypical case: collapsed.  Attempt to
						//                    .    reintroduce ignoring H(K+1,K) and H(K+2,K).
						//                    .    If the fill resulting from the new
						//                    .    reflector is too large, then abandon it.
						//                    .    Otherwise, use the new one. ====
						Zlaqr1(3, h.Off(k, k), s.Get(2*m-1-1), s.Get(2*m-1), vt)
						alpha = vt.Get(0)
						alpha, *vt.GetPtr(0) = Zlarfg(3, alpha, vt.Off(1), 1)
						refsum = vt.GetConj(0) * (h.Get(k, k-1) + vt.GetConj(1)*h.Get(k+2-1, k-1))

						if cabs1(h.Get(k+2-1, k-1)-refsum*vt.Get(1))+cabs1(refsum*vt.Get(2)) > ulp*(cabs1(h.Get(k-1, k-1))+cabs1(h.Get(k, k))+cabs1(h.Get(k+2-1, k+2-1))) {
							//                       ==== Starting a new bulge here would
							//                       .    create non-negligible fill.  Use
							//                       .    the old one with trepidation. ====
							h.Set(k, k-1, beta)
							h.Set(k+2-1, k-1, zero)
							h.Set(k+3-1, k-1, zero)
						} else {
							//                       ==== Stating a new bulge here would
							//                       .    create only negligible fill.
							//                       .    Replace the old reflector with
							//                       .    the new one. ====
							h.Set(k, k-1, h.Get(k, k-1)-refsum)
							h.Set(k+2-1, k-1, zero)
							h.Set(k+3-1, k-1, zero)
							v.Set(0, m-1, vt.Get(0))
							v.Set(1, m-1, vt.Get(1))
							v.Set(2, m-1, vt.Get(2))
						}
					}
				}
			}

			//           ==== Generate a 2-by-2 reflection, if needed. ====
			k = krcol + 3*(m22-1)
			if bmp22 {
				if k == ktop-1 {
					Zlaqr1(2, h.Off(k, k), s.Get(2*m22-1-1), s.Get(2*m22-1), v.Off(0, m22-1).CVector())
					beta = v.Get(0, m22-1)
					beta, *v.GetPtr(0, m22-1) = Zlarfg(2, beta, v.Off(1, m22-1).CVector(), 1)
				} else {
					beta = h.Get(k, k-1)
					v.Set(1, m22-1, h.Get(k+2-1, k-1))
					beta, *v.GetPtr(0, m22-1) = Zlarfg(2, beta, v.Off(1, m22-1).CVector(), 1)
					h.Set(k, k-1, beta)
					h.Set(k+2-1, k-1, zero)
				}
			}

			//           ==== Multiply H by reflections from the left ====
			if accum {
				jbot = min(ndcol, kbot)
			} else if wantt {
				jbot = n
			} else {
				jbot = kbot
			}
			for j = max(ktop, krcol); j <= jbot; j++ {
				mend = min(mbot, (j-krcol+2)/3)
				for m = mtop; m <= mend; m++ {
					k = krcol + 3*(m-1)
					refsum = v.GetConj(0, m-1) * (h.Get(k, j-1) + v.GetConj(1, m-1)*h.Get(k+2-1, j-1) + v.GetConj(2, m-1)*h.Get(k+3-1, j-1))
					h.Set(k, j-1, h.Get(k, j-1)-refsum)
					h.Set(k+2-1, j-1, h.Get(k+2-1, j-1)-refsum*v.Get(1, m-1))
					h.Set(k+3-1, j-1, h.Get(k+3-1, j-1)-refsum*v.Get(2, m-1))
				}
			}
			if bmp22 {
				k = krcol + 3*(m22-1)
				for j = max(k+1, ktop); j <= jbot; j++ {
					refsum = v.GetConj(0, m22-1) * (h.Get(k, j-1) + v.GetConj(1, m22-1)*h.Get(k+2-1, j-1))
					h.Set(k, j-1, h.Get(k, j-1)-refsum)
					h.Set(k+2-1, j-1, h.Get(k+2-1, j-1)-refsum*v.Get(1, m22-1))
				}
			}

			//           ==== Multiply H by reflections from the right.
			//           .    Delay filling in the last row until the
			//           .    vigilant deflation check is complete. ====
			if accum {
				jtop = max(ktop, incol)
			} else if wantt {
				jtop = 1
			} else {
				jtop = ktop
			}
			for m = mtop; m <= mbot; m++ {
				if v.Get(0, m-1) != zero {
					k = krcol + 3*(m-1)
					for j = jtop; j <= min(kbot, k+3); j++ {
						refsum = v.Get(0, m-1) * (h.Get(j-1, k) + v.Get(1, m-1)*h.Get(j-1, k+2-1) + v.Get(2, m-1)*h.Get(j-1, k+3-1))
						h.Set(j-1, k, h.Get(j-1, k)-refsum)
						h.Set(j-1, k+2-1, h.Get(j-1, k+2-1)-refsum*v.GetConj(1, m-1))
						h.Set(j-1, k+3-1, h.Get(j-1, k+3-1)-refsum*v.GetConj(2, m-1))
					}

					if accum {
						//                    ==== Accumulate U. (If necessary, update Z later
						//                    .    with with an efficient matrix-matrix
						//                    .    multiply.) ====
						kms = k - incol
						for j = max(1, ktop-incol); j <= kdu; j++ {
							refsum = v.Get(0, m-1) * (u.Get(j-1, kms) + v.Get(1, m-1)*u.Get(j-1, kms+2-1) + v.Get(2, m-1)*u.Get(j-1, kms+3-1))
							u.Set(j-1, kms, u.Get(j-1, kms)-refsum)
							u.Set(j-1, kms+2-1, u.Get(j-1, kms+2-1)-refsum*v.GetConj(1, m-1))
							u.Set(j-1, kms+3-1, u.Get(j-1, kms+3-1)-refsum*v.GetConj(2, m-1))
						}
					} else if wantz {
						//                    ==== U is not accumulated, so update Z
						//                    .    now by multiplying by reflections
						//                    .    from the right. ====
						for j = iloz; j <= ihiz; j++ {
							refsum = v.Get(0, m-1) * (z.Get(j-1, k) + v.Get(1, m-1)*z.Get(j-1, k+2-1) + v.Get(2, m-1)*z.Get(j-1, k+3-1))
							z.Set(j-1, k, z.Get(j-1, k)-refsum)
							z.Set(j-1, k+2-1, z.Get(j-1, k+2-1)-refsum*v.GetConj(1, m-1))
							z.Set(j-1, k+3-1, z.Get(j-1, k+3-1)-refsum*v.GetConj(2, m-1))
						}
					}
				}
			}

			//           ==== Special case: 2-by-2 reflection (if needed) ====
			k = krcol + 3*(m22-1)
			if bmp22 {
				if v.Get(0, m22-1) != zero {
					for j = jtop; j <= min(kbot, k+3); j++ {
						refsum = v.Get(0, m22-1) * (h.Get(j-1, k) + v.Get(1, m22-1)*h.Get(j-1, k+2-1))
						h.Set(j-1, k, h.Get(j-1, k)-refsum)
						h.Set(j-1, k+2-1, h.Get(j-1, k+2-1)-refsum*v.GetConj(1, m22-1))
					}

					if accum {
						kms = k - incol
						for j = max(1, ktop-incol); j <= kdu; j++ {
							refsum = v.Get(0, m22-1) * (u.Get(j-1, kms) + v.Get(1, m22-1)*u.Get(j-1, kms+2-1))
							u.Set(j-1, kms, u.Get(j-1, kms)-refsum)
							u.Set(j-1, kms+2-1, u.Get(j-1, kms+2-1)-refsum*v.GetConj(1, m22-1))
						}
					} else if wantz {
						for j = iloz; j <= ihiz; j++ {
							refsum = v.Get(0, m22-1) * (z.Get(j-1, k) + v.Get(1, m22-1)*z.Get(j-1, k+2-1))
							z.Set(j-1, k, z.Get(j-1, k)-refsum)
							z.Set(j-1, k+2-1, z.Get(j-1, k+2-1)-refsum*v.GetConj(1, m22-1))
						}
					}
				}
			}

			//           ==== Vigilant deflation check ====
			mstart = mtop
			if krcol+3*(mstart-1) < ktop {
				mstart = mstart + 1
			}
			mend = mbot
			if bmp22 {
				mend = mend + 1
			}
			if krcol == kbot-2 {
				mend = mend + 1
			}
			for m = mstart; m <= mend; m++ {
				k = min(kbot-1, krcol+3*(m-1))

				//              ==== The following convergence test requires that
				//              .    the tradition small-compared-to-nearby-diagonals
				//              .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
				//              .    criteria both be satisfied.  The latter improves
				//              .    accuracy in some examples. Falling back on an
				//              .    alternate convergence criterion when TST1 or TST2
				//              .    is zero (as done here) is traditional but probably
				//              .    unnecessary. ====
				if h.Get(k, k-1) != zero {
					tst1 = cabs1(h.Get(k-1, k-1)) + cabs1(h.Get(k, k))
					if tst1 == rzero {
						if k >= ktop+1 {
							tst1 = tst1 + cabs1(h.Get(k-1, k-1-1))
						}
						if k >= ktop+2 {
							tst1 = tst1 + cabs1(h.Get(k-1, k-2-1))
						}
						if k >= ktop+3 {
							tst1 = tst1 + cabs1(h.Get(k-1, k-3-1))
						}
						if k <= kbot-2 {
							tst1 = tst1 + cabs1(h.Get(k+2-1, k))
						}
						if k <= kbot-3 {
							tst1 = tst1 + cabs1(h.Get(k+3-1, k))
						}
						if k <= kbot-4 {
							tst1 = tst1 + cabs1(h.Get(k+4-1, k))
						}
					}
					if cabs1(h.Get(k, k-1)) <= math.Max(smlnum, ulp*tst1) {
						h12 = math.Max(cabs1(h.Get(k, k-1)), cabs1(h.Get(k-1, k)))
						h21 = math.Min(cabs1(h.Get(k, k-1)), cabs1(h.Get(k-1, k)))
						h11 = math.Max(cabs1(h.Get(k, k)), cabs1(h.Get(k-1, k-1)-h.Get(k, k)))
						h22 = math.Min(cabs1(h.Get(k, k)), cabs1(h.Get(k-1, k-1)-h.Get(k, k)))
						scl = h11 + h12
						tst2 = h22 * (h11 / scl)

						if tst2 == rzero || h21*(h12/scl) <= math.Max(smlnum, ulp*tst2) {
							h.Set(k, k-1, zero)
						}
					}
				}
			}

			//           ==== Fill in the last row of each bulge. ====
			mend = min(nbmps, (kbot-krcol-1)/3)
			for m = mtop; m <= mend; m++ {
				k = krcol + 3*(m-1)
				refsum = v.Get(0, m-1) * v.Get(2, m-1) * h.Get(k+4-1, k+3-1)
				h.Set(k+4-1, k, -refsum)
				h.Set(k+4-1, k+2-1, -refsum*v.GetConj(1, m-1))
				h.Set(k+4-1, k+3-1, h.Get(k+4-1, k+3-1)-refsum*v.GetConj(2, m-1))
			}

			//           ==== End of near-the-diagonal bulge chase. ====
		}

		//        ==== Use U (if accumulated) to update far-from-diagonal
		//        .    entries in H.  If required, use U to update Z as
		//        .    well. ====
		if accum {
			if wantt {
				jtop = 1
				jbot = n
			} else {
				jtop = ktop
				jbot = kbot
			}
			if (!blk22) || (incol < ktop) || (ndcol > kbot) || (ns <= 2) {
				//              ==== Updates not exploiting the 2-by-2 block
				//              .    structure of U.  K1 and NU keep track of
				//              .    the location and size of U in the special
				//              .    cases of introducing bulges and chasing
				//              .    bulges off the bottom.  In these special
				//              .    cases and in case the number of shifts
				//              .    is NS = 2, there is no 2-by-2 block
				//              .    structure to exploit.  ====
				k1 = max(1, ktop-incol)
				nu = (kdu - max(0, ndcol-kbot)) - k1 + 1

				//              ==== Horizontal Multiply ====
				for jcol = min(ndcol, kbot) + 1; jcol <= jbot; jcol += nh {
					jlen = min(nh, jbot-jcol+1)
					if err = wh.Gemm(ConjTrans, NoTrans, nu, jlen, nu, one, u.Off(k1-1, k1-1), h.Off(incol+k1-1, jcol-1), zero); err != nil {
						panic(err)
					}
					Zlacpy(Full, nu, jlen, wh, h.Off(incol+k1-1, jcol-1))
				}

				//              ==== Vertical multiply ====
				for jrow = jtop; jrow <= max(ktop, incol)-1; jrow += nv {
					jlen = min(nv, max(ktop, incol)-jrow)
					if err = wv.Gemm(NoTrans, NoTrans, jlen, nu, nu, one, h.Off(jrow-1, incol+k1-1), u.Off(k1-1, k1-1), zero); err != nil {
						panic(err)
					}
					Zlacpy(Full, jlen, nu, wv, h.Off(jrow-1, incol+k1-1))
				}

				//              ==== Z multiply (also vertical) ====
				if wantz {
					for jrow = iloz; jrow <= ihiz; jrow += nv {
						jlen = min(nv, ihiz-jrow+1)
						if err = wv.Gemm(NoTrans, NoTrans, jlen, nu, nu, one, z.Off(jrow-1, incol+k1-1), u.Off(k1-1, k1-1), zero); err != nil {
							panic(err)
						}
						Zlacpy(Full, jlen, nu, wv, z.Off(jrow-1, incol+k1-1))
					}
				}
			} else {
				//              ==== Updates exploiting U's 2-by-2 block structure.
				//              .    (I2, I4, J2, J4 are the last rows and columns
				//              .    of the blocks.) ====
				i2 = (kdu + 1) / 2
				i4 = kdu
				j2 = i4 - i2
				j4 = kdu

				//              ==== KZS and KNZ deal with the band of zeros
				//              .    along the diagonal of one of the triangular
				//              .    blocks. ====
				kzs = (j4 - j2) - (ns + 1)
				knz = ns + 1

				//              ==== Horizontal multiply ====
				for jcol = min(ndcol, kbot) + 1; jcol <= jbot; jcol += nh {
					jlen = min(nh, jbot-jcol+1)

					//                 ==== Copy bottom of H to top+KZS of scratch ====
					//                  (The first KZS rows get multiplied by zero.) ====
					Zlacpy(Full, knz, jlen, h.Off(incol+1+j2-1, jcol-1), wh.Off(kzs, 0))

					//                 ==== Multiply by U21**H ====
					Zlaset(Full, kzs, jlen, zero, zero, wh)
					if err = wh.Off(kzs, 0).Trmm(Left, Upper, ConjTrans, NonUnit, knz, jlen, one, u.Off(j2, 1+kzs-1)); err != nil {
						panic(err)
					}

					//                 ==== Multiply top of H by U11**H ====
					if err = wh.Gemm(ConjTrans, NoTrans, i2, jlen, j2, one, u, h.Off(incol, jcol-1), one); err != nil {
						panic(err)
					}

					//                 ==== Copy top of H to bottom of WH ====
					Zlacpy(Full, j2, jlen, h.Off(incol, jcol-1), wh.Off(i2, 0))

					//                 ==== Multiply by U21**H ====
					if err = wh.Off(i2, 0).Trmm(Left, Lower, ConjTrans, NonUnit, j2, jlen, one, u.Off(0, i2)); err != nil {
						panic(err)
					}

					//                 ==== Multiply by U22 ====
					if err = wh.Off(i2, 0).Gemm(ConjTrans, NoTrans, i4-i2, jlen, j4-j2, one, u.Off(j2, i2), h.Off(incol+1+j2-1, jcol-1), one); err != nil {
						panic(err)
					}

					//                 ==== Copy it back ====
					Zlacpy(Full, kdu, jlen, wh, h.Off(incol, jcol-1))
				}

				//              ==== Vertical multiply ====
				for jrow = jtop; jrow <= max(incol, ktop)-1; jrow += nv {
					jlen = min(nv, max(incol, ktop)-jrow)

					//                 ==== Copy right of H to scratch (the first KZS
					//                 .    columns get multiplied by zero) ====
					Zlacpy(Full, jlen, knz, h.Off(jrow-1, incol+1+j2-1), wv.Off(0, 1+kzs-1))

					//                 ==== Multiply by U21 ====
					Zlaset(Full, jlen, kzs, zero, zero, wv)
					if err = wv.Off(0, 1+kzs-1).Trmm(Right, Upper, NoTrans, NonUnit, jlen, knz, one, u.Off(j2, 1+kzs-1)); err != nil {
						panic(err)
					}

					//                 ==== Multiply by U11 ====
					if err = wv.Gemm(NoTrans, NoTrans, jlen, i2, j2, one, h.Off(jrow-1, incol), u, one); err != nil {
						panic(err)
					}

					//                 ==== Copy left of H to right of scratch ====
					Zlacpy(Full, jlen, j2, h.Off(jrow-1, incol), wv.Off(0, 1+i2-1))

					//                 ==== Multiply by U21 ====
					if err = wv.Off(0, 1+i2-1).Trmm(Right, Lower, NoTrans, NonUnit, jlen, i4-i2, one, u.Off(0, i2)); err != nil {
						panic(err)
					}

					//                 ==== Multiply by U22 ====
					if err = wv.Off(0, 1+i2-1).Gemm(NoTrans, NoTrans, jlen, i4-i2, j4-j2, one, h.Off(jrow-1, incol+1+j2-1), u.Off(j2, i2), one); err != nil {
						panic(err)
					}

					//                 ==== Copy it back ====
					Zlacpy(Full, jlen, kdu, wv, h.Off(jrow-1, incol))
				}

				//              ==== Multiply Z (also vertical) ====
				if wantz {
					for jrow = iloz; jrow <= ihiz; jrow += nv {
						jlen = min(nv, ihiz-jrow+1)

						//                    ==== Copy right of Z to left of scratch (first
						//                    .     KZS columns get multiplied by zero) ====
						Zlacpy(Full, jlen, knz, z.Off(jrow-1, incol+1+j2-1), wv.Off(0, 1+kzs-1))

						//                    ==== Multiply by U12 ====
						Zlaset(Full, jlen, kzs, zero, zero, wv)
						if err = wv.Off(0, 1+kzs-1).Trmm(Right, Upper, NoTrans, NonUnit, jlen, knz, one, u.Off(j2, 1+kzs-1)); err != nil {
							panic(err)
						}

						//                    ==== Multiply by U11 ====
						if err = wv.Gemm(NoTrans, NoTrans, jlen, i2, j2, one, z.Off(jrow-1, incol), u, one); err != nil {
							panic(err)
						}

						//                    ==== Copy left of Z to right of scratch ====
						Zlacpy(Full, jlen, j2, z.Off(jrow-1, incol), wv.Off(0, 1+i2-1))

						//                    ==== Multiply by U21 ====
						if err = wv.Off(0, 1+i2-1).Trmm(Right, Lower, NoTrans, NonUnit, jlen, i4-i2, one, u.Off(0, i2)); err != nil {
							panic(err)
						}

						//                    ==== Multiply by U22 ====
						if err = wv.Off(0, 1+i2-1).Gemm(NoTrans, NoTrans, jlen, i4-i2, j4-j2, one, z.Off(jrow-1, incol+1+j2-1), u.Off(j2, i2), one); err != nil {
							panic(err)
						}

						//                    ==== Copy the result back to Z ====
						Zlacpy(Full, jlen, kdu, wv, z.Off(jrow-1, incol))
					}
				}
			}
		}
	}
}
