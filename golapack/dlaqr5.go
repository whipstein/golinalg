package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlaqr5 called by DLAQR0, performs a
//    single small-bulge multi-shift QR sweep.
func Dlaqr5(wantt, wantz bool, kacc22, n, ktop, kbot, nshfts *int, sr, si *mat.Vector, h *mat.Matrix, ldh, iloz, ihiz *int, z *mat.Matrix, ldz *int, v *mat.Matrix, ldv *int, u *mat.Matrix, ldu, nv *int, wv *mat.Matrix, ldwv, nh *int, wh *mat.Matrix, ldwh *int) {
	var accum, blk22, bmp22 bool
	var alpha, beta, h11, h12, h21, h22, one, refsum, safmax, safmin, scl, smlnum, swap, tst1, tst2, ulp, zero float64
	var i, i2, i4, incol, j, j2, j4, jbot, jcol, jlen, jrow, jtop, k, k1, kdu, kms, knz, krcol, kzs, m, m22, mbot, mend, mstart, mtop, nbmps, ndcol, ns, nu int
	var err error
	_ = err

	vt := vf(3)

	zero = 0.0
	one = 1.0

	//     ==== If there are no shifts, then there is nothing to do. ====
	if (*nshfts) < 2 {
		return
	}

	//     ==== If the active block is empty or 1-by-1, then there
	//     .    is nothing to do. ====
	if (*ktop) >= (*kbot) {
		return
	}

	//     ==== Shuffle shifts into pairs of real shifts and pairs
	//     .    of complex conjugate shifts assuming complex
	//     .    conjugate shifts are already adjacent to one
	//     .    another. ====
	for i = 1; i <= (*nshfts)-2; i += 2 {
		if si.Get(i-1) != -si.Get(i+1-1) {

			swap = sr.Get(i - 1)
			sr.Set(i-1, sr.Get(i+1-1))
			sr.Set(i+1-1, sr.Get(i+2-1))
			sr.Set(i+2-1, swap)

			swap = si.Get(i - 1)
			si.Set(i-1, si.Get(i+1-1))
			si.Set(i+1-1, si.Get(i+2-1))
			si.Set(i+2-1, swap)
		}
	}

	//     ==== NSHFTS is supposed to be even, but if it is odd,
	//     .    then simply reduce it by one.  The shuffle above
	//     .    ensures that the dropped shift is real and that
	//     .    the remaining shifts are paired. ====
	ns = (*nshfts) - (*nshfts % 2)

	//     ==== Machine constants for deflation ====
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	Dlabad(&safmin, &safmax)
	ulp = Dlamch(Precision)
	smlnum = safmin * (float64(*n) / ulp)

	//     ==== Use accumulated reflections to update far-from-diagonal
	//     .    entries ? ====
	accum = ((*kacc22) == 1) || ((*kacc22) == 2)

	//     ==== If so, exploit the 2-by-2 block structure? ====
	blk22 = (ns > 2) && ((*kacc22) == 2)

	//     ==== clear trash ====
	if (*ktop)+2 <= (*kbot) {
		h.Set((*ktop)+2-1, (*ktop)-1, zero)
	}

	//     ==== NBMPS = number of 2-shift bulges in the chain ====
	nbmps = ns / 2

	//     ==== KDU = width of slab ====
	kdu = 6*nbmps - 3

	//     ==== Create and chase chains of NBMPS bulges ====
	for incol = 3*(1-nbmps) + (*ktop) - 1; incol <= (*kbot)-2; incol += 3*nbmps - 2 {
		ndcol = incol + kdu
		if accum {
			Dlaset('A', &kdu, &kdu, &zero, &one, u, ldu)
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
		for krcol = incol; krcol <= minint(incol+3*nbmps-3, (*kbot)-2); krcol++ {
			//           ==== Bulges number MTOP to MBOT are active double implicit
			//           .    shift bulges.  There may or may not also be small
			//           .    2-by-2 bulge, if there is room.  The inactive bulges
			//           .    (if any) must wait until the active bulges have moved
			//           .    down the diagonal to make room.  The phantom matrix
			//           .    paradigm described above helps keep track.  ====
			mtop = maxint(1, (((*ktop)-1)-krcol+2)/3+1)
			mbot = minint(nbmps, ((*kbot)-krcol)/3)
			m22 = mbot + 1
			bmp22 = (mbot < nbmps) && (krcol+3*(m22-1)) == ((*kbot)-2)

			//           ==== Generate reflections to chase the chain right
			//           .    one column.  (The minimum value of K is KTOP-1.) ====
			for m = mtop; m <= mbot; m++ {
				k = krcol + 3*(m-1)
				if k == (*ktop)-1 {
					Dlaqr1(func() *int { y := 3; return &y }(), h.Off((*ktop)-1, (*ktop)-1), ldh, sr.GetPtr(2*m-1-1), si.GetPtr(2*m-1-1), sr.GetPtr(2*m-1), si.GetPtr(2*m-1), v.Vector(0, m-1))
					alpha = v.Get(0, m-1)
					Dlarfg(func() *int { y := 3; return &y }(), &alpha, v.Vector(1, m-1), func() *int { y := 1; return &y }(), v.GetPtr(0, m-1))
				} else {
					beta = h.Get(k+1-1, k-1)
					v.Set(1, m-1, h.Get(k+2-1, k-1))
					v.Set(2, m-1, h.Get(k+3-1, k-1))
					Dlarfg(func() *int { y := 3; return &y }(), &beta, v.Vector(1, m-1), func() *int { y := 1; return &y }(), v.GetPtr(0, m-1))

					//                 ==== A Bulge may collapse because of vigilant
					//                 .    deflation or destructive underflow.  In the
					//                 .    underflow case, try the two-small-subdiagonals
					//                 .    trick to try to reinflate the bulge.  ====
					if h.Get(k+3-1, k-1) != zero || h.Get(k+3-1, k+1-1) != zero || h.Get(k+3-1, k+2-1) == zero {
						//                    ==== Typical case: not collapsed (yet). ====
						h.Set(k+1-1, k-1, beta)
						h.Set(k+2-1, k-1, zero)
						h.Set(k+3-1, k-1, zero)
					} else {
						//                    ==== Atypical case: collapsed.  Attempt to
						//                    .    reintroduce ignoring H(K+1,K) and H(K+2,K).
						//                    .    If the fill resulting from the new
						//                    .    reflector is too large, then abandon it.
						//                    .    Otherwise, use the new one. ====
						Dlaqr1(func() *int { y := 3; return &y }(), h.Off(k+1-1, k+1-1), ldh, sr.GetPtr(2*m-1-1), si.GetPtr(2*m-1-1), sr.GetPtr(2*m-1), si.GetPtr(2*m-1), vt)
						alpha = vt.Get(0)
						Dlarfg(func() *int { y := 3; return &y }(), &alpha, vt.Off(1), func() *int { y := 1; return &y }(), vt.GetPtr(0))
						refsum = vt.Get(0) * (h.Get(k+1-1, k-1) + vt.Get(1)*h.Get(k+2-1, k-1))

						if math.Abs(h.Get(k+2-1, k-1)-refsum*vt.Get(1))+math.Abs(refsum*vt.Get(2)) > ulp*(math.Abs(h.Get(k-1, k-1))+math.Abs(h.Get(k+1-1, k+1-1))+math.Abs(h.Get(k+2-1, k+2-1))) {
							//                       ==== Starting a new bulge here would
							//                       .    create non-negligible fill.  Use
							//                       .    the old one with trepidation. ====
							h.Set(k+1-1, k-1, beta)
							h.Set(k+2-1, k-1, zero)
							h.Set(k+3-1, k-1, zero)
						} else {
							//                       ==== Stating a new bulge here would
							//                       .    create only negligible fill.
							//                       .    Replace the old reflector with
							//                       .    the new one. ====
							h.Set(k+1-1, k-1, h.Get(k+1-1, k-1)-refsum)
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
				if k == (*ktop)-1 {
					Dlaqr1(func() *int { y := 2; return &y }(), h.Off(k+1-1, k+1-1), ldh, sr.GetPtr(2*m22-1-1), si.GetPtr(2*m22-1-1), sr.GetPtr(2*m22-1), si.GetPtr(2*m22-1), v.Vector(0, m22-1))
					beta = v.Get(0, m22-1)
					Dlarfg(func() *int { y := 2; return &y }(), &beta, v.Vector(1, m22-1), func() *int { y := 1; return &y }(), v.GetPtr(0, m22-1))
				} else {
					beta = h.Get(k+1-1, k-1)
					v.Set(1, m22-1, h.Get(k+2-1, k-1))
					Dlarfg(func() *int { y := 2; return &y }(), &beta, v.Vector(1, m22-1), func() *int { y := 1; return &y }(), v.GetPtr(0, m22-1))
					h.Set(k+1-1, k-1, beta)
					h.Set(k+2-1, k-1, zero)
				}
			}

			//           ==== Multiply H by reflections from the left ====
			if accum {
				jbot = minint(ndcol, *kbot)
			} else if wantt {
				jbot = (*n)
			} else {
				jbot = (*kbot)
			}
			for j = maxint(*ktop, krcol); j <= jbot; j++ {
				mend = minint(mbot, (j-krcol+2)/3)
				for m = mtop; m <= mend; m++ {
					k = krcol + 3*(m-1)
					refsum = v.Get(0, m-1) * (h.Get(k+1-1, j-1) + v.Get(1, m-1)*h.Get(k+2-1, j-1) + v.Get(2, m-1)*h.Get(k+3-1, j-1))
					h.Set(k+1-1, j-1, h.Get(k+1-1, j-1)-refsum)
					h.Set(k+2-1, j-1, h.Get(k+2-1, j-1)-refsum*v.Get(1, m-1))
					h.Set(k+3-1, j-1, h.Get(k+3-1, j-1)-refsum*v.Get(2, m-1))
				}
			}
			if bmp22 {
				k = krcol + 3*(m22-1)
				for j = maxint(k+1, *ktop); j <= jbot; j++ {
					refsum = v.Get(0, m22-1) * (h.Get(k+1-1, j-1) + v.Get(1, m22-1)*h.Get(k+2-1, j-1))
					h.Set(k+1-1, j-1, h.Get(k+1-1, j-1)-refsum)
					h.Set(k+2-1, j-1, h.Get(k+2-1, j-1)-refsum*v.Get(1, m22-1))
				}
			}

			//           ==== Multiply H by reflections from the right.
			//           .    Delay filling in the last row until the
			//           .    vigilant deflation check is complete. ====
			if accum {
				jtop = maxint(*ktop, incol)
			} else if wantt {
				jtop = 1
			} else {
				jtop = (*ktop)
			}
			for m = mtop; m <= mbot; m++ {
				if v.Get(0, m-1) != zero {
					k = krcol + 3*(m-1)
					for j = jtop; j <= minint(*kbot, k+3); j++ {
						refsum = v.Get(0, m-1) * (h.Get(j-1, k+1-1) + v.Get(1, m-1)*h.Get(j-1, k+2-1) + v.Get(2, m-1)*h.Get(j-1, k+3-1))
						h.Set(j-1, k+1-1, h.Get(j-1, k+1-1)-refsum)
						h.Set(j-1, k+2-1, h.Get(j-1, k+2-1)-refsum*v.Get(1, m-1))
						h.Set(j-1, k+3-1, h.Get(j-1, k+3-1)-refsum*v.Get(2, m-1))
					}

					if accum {
						//                    ==== Accumulate U. (If necessary, update Z later
						//                    .    with with an efficient matrix-matrix
						//                    .    multiply.) ====
						kms = k - incol
						for j = maxint(1, (*ktop)-incol); j <= kdu; j++ {
							refsum = v.Get(0, m-1) * (u.Get(j-1, kms+1-1) + v.Get(1, m-1)*u.Get(j-1, kms+2-1) + v.Get(2, m-1)*u.Get(j-1, kms+3-1))
							u.Set(j-1, kms+1-1, u.Get(j-1, kms+1-1)-refsum)
							u.Set(j-1, kms+2-1, u.Get(j-1, kms+2-1)-refsum*v.Get(1, m-1))
							u.Set(j-1, kms+3-1, u.Get(j-1, kms+3-1)-refsum*v.Get(2, m-1))
						}
					} else if wantz {
						//                    ==== U is not accumulated, so update Z
						//                    .    now by multiplying by reflections
						//                    .    from the right. ====
						for j = (*iloz); j <= (*ihiz); j++ {
							refsum = v.Get(0, m-1) * (z.Get(j-1, k+1-1) + v.Get(1, m-1)*z.Get(j-1, k+2-1) + v.Get(2, m-1)*z.Get(j-1, k+3-1))
							z.Set(j-1, k+1-1, z.Get(j-1, k+1-1)-refsum)
							z.Set(j-1, k+2-1, z.Get(j-1, k+2-1)-refsum*v.Get(1, m-1))
							z.Set(j-1, k+3-1, z.Get(j-1, k+3-1)-refsum*v.Get(2, m-1))
						}
					}
				}
			}

			//           ==== Special case: 2-by-2 reflection (if needed) ====
			k = krcol + 3*(m22-1)
			if bmp22 {
				if v.Get(0, m22-1) != zero {
					for j = jtop; j <= minint(*kbot, k+3); j++ {
						refsum = v.Get(0, m22-1) * (h.Get(j-1, k+1-1) + v.Get(1, m22-1)*h.Get(j-1, k+2-1))
						h.Set(j-1, k+1-1, h.Get(j-1, k+1-1)-refsum)
						h.Set(j-1, k+2-1, h.Get(j-1, k+2-1)-refsum*v.Get(1, m22-1))
					}

					if accum {
						kms = k - incol
						for j = maxint(1, (*ktop)-incol); j <= kdu; j++ {
							refsum = v.Get(0, m22-1) * (u.Get(j-1, kms+1-1) + v.Get(1, m22-1)*u.Get(j-1, kms+2-1))
							u.Set(j-1, kms+1-1, u.Get(j-1, kms+1-1)-refsum)
							u.Set(j-1, kms+2-1, u.Get(j-1, kms+2-1)-refsum*v.Get(1, m22-1))
						}
					} else if wantz {
						for j = (*iloz); j <= (*ihiz); j++ {
							refsum = v.Get(0, m22-1) * (z.Get(j-1, k+1-1) + v.Get(1, m22-1)*z.Get(j-1, k+2-1))
							z.Set(j-1, k+1-1, z.Get(j-1, k+1-1)-refsum)
							z.Set(j-1, k+2-1, z.Get(j-1, k+2-1)-refsum*v.Get(1, m22-1))
						}
					}
				}
			}

			//           ==== Vigilant deflation check ====
			mstart = mtop
			if krcol+3*(mstart-1) < (*ktop) {
				mstart = mstart + 1
			}
			mend = mbot
			if bmp22 {
				mend = mend + 1
			}
			if krcol == (*kbot)-2 {
				mend = mend + 1
			}
			for m = mstart; m <= mend; m++ {
				k = minint((*kbot)-1, krcol+3*(m-1))

				//              ==== The following convergence test requires that
				//              .    the tradition small-compared-to-nearby-diagonals
				//              .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
				//              .    criteria both be satisfied.  The latter improves
				//              .    accuracy in some examples. Falling back on an
				//              .    alternate convergence criterion when TST1 or TST2
				//              .    is zero (as done here) is traditional but probably
				//              .    unnecessary. ====
				if h.Get(k+1-1, k-1) != zero {
					tst1 = math.Abs(h.Get(k-1, k-1)) + math.Abs(h.Get(k+1-1, k+1-1))
					if tst1 == zero {
						if k >= (*ktop)+1 {
							tst1 = tst1 + math.Abs(h.Get(k-1, k-1-1))
						}
						if k >= (*ktop)+2 {
							tst1 = tst1 + math.Abs(h.Get(k-1, k-2-1))
						}
						if k >= (*ktop)+3 {
							tst1 = tst1 + math.Abs(h.Get(k-1, k-3-1))
						}
						if k <= (*kbot)-2 {
							tst1 = tst1 + math.Abs(h.Get(k+2-1, k+1-1))
						}
						if k <= (*kbot)-3 {
							tst1 = tst1 + math.Abs(h.Get(k+3-1, k+1-1))
						}
						if k <= (*kbot)-4 {
							tst1 = tst1 + math.Abs(h.Get(k+4-1, k+1-1))
						}
					}
					if math.Abs(h.Get(k+1-1, k-1)) <= maxf64(smlnum, ulp*tst1) {
						h12 = maxf64(math.Abs(h.Get(k+1-1, k-1)), math.Abs(h.Get(k-1, k+1-1)))
						h21 = minf64(math.Abs(h.Get(k+1-1, k-1)), math.Abs(h.Get(k-1, k+1-1)))
						h11 = maxf64(math.Abs(h.Get(k+1-1, k+1-1)), math.Abs(h.Get(k-1, k-1)-h.Get(k+1-1, k+1-1)))
						h22 = minf64(math.Abs(h.Get(k+1-1, k+1-1)), math.Abs(h.Get(k-1, k-1)-h.Get(k+1-1, k+1-1)))
						scl = h11 + h12
						tst2 = h22 * (h11 / scl)

						if tst2 == zero || h21*(h12/scl) <= maxf64(smlnum, ulp*tst2) {
							h.Set(k+1-1, k-1, zero)
						}
					}
				}
			}

			//           ==== Fill in the last row of each bulge. ====
			mend = minint(nbmps, ((*kbot)-krcol-1)/3)
			for m = mtop; m <= mend; m++ {
				k = krcol + 3*(m-1)
				refsum = v.Get(0, m-1) * v.Get(2, m-1) * h.Get(k+4-1, k+3-1)
				h.Set(k+4-1, k+1-1, -refsum)
				h.Set(k+4-1, k+2-1, -refsum*v.Get(1, m-1))
				h.Set(k+4-1, k+3-1, h.Get(k+4-1, k+3-1)-refsum*v.Get(2, m-1))
			}

			//           ==== End of near-the-diagonal bulge chase. ====
		}

		//        ==== Use U (if accumulated) to update far-from-diagonal
		//        .    entries in H.  If required, use U to update Z as
		//        .    well. ====
		if accum {
			if wantt {
				jtop = 1
				jbot = (*n)
			} else {
				jtop = (*ktop)
				jbot = (*kbot)
			}
			if (!blk22) || (incol < (*ktop)) || (ndcol > (*kbot)) || (ns <= 2) {
				//              ==== Updates not exploiting the 2-by-2 block
				//              .    structure of U.  K1 and NU keep track of
				//              .    the location and size of U in the special
				//              .    cases of introducing bulges and chasing
				//              .    bulges off the bottom.  In these special
				//              .    cases and in case the number of shifts
				//              .    is NS = 2, there is no 2-by-2 block
				//              .    structure to exploit.  ====
				k1 = maxint(1, (*ktop)-incol)
				nu = (kdu - maxint(0, ndcol-(*kbot))) - k1 + 1

				//              ==== Horizontal Multiply ====
				for jcol = minint(ndcol, *kbot) + 1; jcol <= jbot; jcol += (*nh) {
					jlen = minint(*nh, jbot-jcol+1)
					err = goblas.Dgemm(ConjTrans, NoTrans, nu, jlen, nu, one, u.Off(k1-1, k1-1), *ldu, h.Off(incol+k1-1, jcol-1), *ldh, zero, wh, *ldwh)
					Dlacpy('A', &nu, &jlen, wh, ldwh, h.Off(incol+k1-1, jcol-1), ldh)
				}

				//              ==== Vertical multiply ====
				for jrow = jtop; jrow <= maxint(*ktop, incol)-1; jrow += (*nv) {
					jlen = minint(*nv, maxint(*ktop, incol)-jrow)
					err = goblas.Dgemm(NoTrans, NoTrans, jlen, nu, nu, one, h.Off(jrow-1, incol+k1-1), *ldh, u.Off(k1-1, k1-1), *ldu, zero, wv, *ldwv)
					Dlacpy('A', &jlen, &nu, wv, ldwv, h.Off(jrow-1, incol+k1-1), ldh)
				}

				//              ==== Z multiply (also vertical) ====
				if wantz {
					for jrow = (*iloz); jrow <= (*ihiz); jrow += (*nv) {
						jlen = minint(*nv, (*ihiz)-jrow+1)
						err = goblas.Dgemm(NoTrans, NoTrans, jlen, nu, nu, one, z.Off(jrow-1, incol+k1-1), *ldz, u.Off(k1-1, k1-1), *ldu, zero, wv, *ldwv)
						Dlacpy('A', &jlen, &nu, wv, ldwv, z.Off(jrow-1, incol+k1-1), ldz)
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
				for jcol = minint(ndcol, *kbot) + 1; jcol <= jbot; jcol += (*nh) {
					jlen = minint(*nh, jbot-jcol+1)

					//                 ==== Copy bottom of H to top+KZS of scratch ====
					//                  (The first KZS rows get multiplied by zero.) ====
					Dlacpy('A', &knz, &jlen, h.Off(incol+1+j2-1, jcol-1), ldh, wh.Off(kzs+1-1, 0), ldwh)

					//                 ==== Multiply by U21**T ====
					Dlaset('A', &kzs, &jlen, &zero, &zero, wh, ldwh)
					err = goblas.Dtrmm(Left, Upper, ConjTrans, NonUnit, knz, jlen, one, u.Off(j2+1-1, 1+kzs-1), *ldu, wh.Off(kzs+1-1, 0), *ldwh)

					//                 ==== Multiply top of H by U11**T ====
					err = goblas.Dgemm(ConjTrans, NoTrans, i2, jlen, j2, one, u, *ldu, h.Off(incol+1-1, jcol-1), *ldh, one, wh, *ldwh)

					//                 ==== Copy top of H to bottom of WH ====
					Dlacpy('A', &j2, &jlen, h.Off(incol+1-1, jcol-1), ldh, wh.Off(i2+1-1, 0), ldwh)

					//                 ==== Multiply by U21**T ====
					err = goblas.Dtrmm(Left, Lower, ConjTrans, NonUnit, j2, jlen, one, u.Off(0, i2+1-1), *ldu, wh.Off(i2+1-1, 0), *ldwh)

					//                 ==== Multiply by U22 ====
					err = goblas.Dgemm(ConjTrans, NoTrans, i4-i2, jlen, j4-j2, one, u.Off(j2+1-1, i2+1-1), *ldu, h.Off(incol+1+j2-1, jcol-1), *ldh, one, wh.Off(i2+1-1, 0), *ldwh)

					//                 ==== Copy it back ====
					Dlacpy('A', &kdu, &jlen, wh, ldwh, h.Off(incol+1-1, jcol-1), ldh)
				}

				//              ==== Vertical multiply ====
				for jrow = jtop; jrow <= maxint(incol, *ktop)-1; jrow += (*nv) {
					jlen = minint(*nv, maxint(incol, *ktop)-jrow)

					//                 ==== Copy right of H to scratch (the first KZS
					//                 .    columns get multiplied by zero) ====
					Dlacpy('A', &jlen, &knz, h.Off(jrow-1, incol+1+j2-1), ldh, wv.Off(0, 1+kzs-1), ldwv)

					//                 ==== Multiply by U21 ====
					Dlaset('A', &jlen, &kzs, &zero, &zero, wv, ldwv)
					err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, jlen, knz, one, u.Off(j2+1-1, 1+kzs-1), *ldu, wv.Off(0, 1+kzs-1), *ldwv)

					//                 ==== Multiply by U11 ====
					err = goblas.Dgemm(NoTrans, NoTrans, jlen, i2, j2, one, h.Off(jrow-1, incol+1-1), *ldh, u, *ldu, one, wv, *ldwv)

					//                 ==== Copy left of H to right of scratch ====
					Dlacpy('A', &jlen, &j2, h.Off(jrow-1, incol+1-1), ldh, wv.Off(0, 1+i2-1), ldwv)

					//                 ==== Multiply by U21 ====
					err = goblas.Dtrmm(Right, Lower, NoTrans, NonUnit, jlen, i4-i2, one, u.Off(0, i2+1-1), *ldu, wv.Off(0, 1+i2-1), *ldwv)

					//                 ==== Multiply by U22 ====
					err = goblas.Dgemm(NoTrans, NoTrans, jlen, i4-i2, j4-j2, one, h.Off(jrow-1, incol+1+j2-1), *ldh, u.Off(j2+1-1, i2+1-1), *ldu, one, wv.Off(0, 1+i2-1), *ldwv)

					//                 ==== Copy it back ====
					Dlacpy('A', &jlen, &kdu, wv, ldwv, h.Off(jrow-1, incol+1-1), ldh)
				}

				//              ==== Multiply Z (also vertical) ====
				if wantz {
					for jrow = (*iloz); jrow <= (*ihiz); jrow += (*nv) {
						jlen = minint(*nv, (*ihiz)-jrow+1)

						//                    ==== Copy right of Z to left of scratch (first
						//                    .     KZS columns get multiplied by zero) ====
						Dlacpy('A', &jlen, &knz, z.Off(jrow-1, incol+1+j2-1), ldz, wv.Off(0, 1+kzs-1), ldwv)

						//                    ==== Multiply by U12 ====
						Dlaset('A', &jlen, &kzs, &zero, &zero, wv, ldwv)
						err = goblas.Dtrmm(Right, Upper, NoTrans, NonUnit, jlen, knz, one, u.Off(j2+1-1, 1+kzs-1), *ldu, wv.Off(0, 1+kzs-1), *ldwv)

						//                    ==== Multiply by U11 ====
						err = goblas.Dgemm(NoTrans, NoTrans, jlen, i2, j2, one, z.Off(jrow-1, incol+1-1), *ldz, u, *ldu, one, wv, *ldwv)

						//                    ==== Copy left of Z to right of scratch ====
						Dlacpy('A', &jlen, &j2, z.Off(jrow-1, incol+1-1), ldz, wv.Off(0, 1+i2-1), ldwv)

						//                    ==== Multiply by U21 ====
						err = goblas.Dtrmm(Right, Lower, NoTrans, NonUnit, jlen, i4-i2, one, u.Off(0, i2+1-1), *ldu, wv.Off(0, 1+i2-1), *ldwv)

						//                    ==== Multiply by U22 ====
						err = goblas.Dgemm(NoTrans, NoTrans, jlen, i4-i2, j4-j2, one, z.Off(jrow-1, incol+1+j2-1), *ldz, u.Off(j2+1-1, i2+1-1), *ldu, one, wv.Off(0, 1+i2-1), *ldwv)

						//                    ==== Copy the result back to Z ====
						Dlacpy('A', &jlen, &kdu, wv, ldwv, z.Off(jrow-1, incol+1-1), ldz)
					}
				}
			}
		}
	}
}
