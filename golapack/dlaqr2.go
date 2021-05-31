package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlaqr2 is identical to DLAQR3 except that it avoids
//    recursion by calling DLAHQR instead of DLAQR4.
//
//    Aggressive early deflation:
//
//    This subroutine accepts as input an upper Hessenberg matrix
//    H and performs an orthogonal similarity transformation
//    designed to detect and deflate fully converged eigenvalues from
//    a trailing principal submatrix.  On output H has been over-
//    written by a new Hessenberg matrix that is a perturbation of
//    an orthogonal similarity transformation of H.  It is to be
//    hoped that the final version of H has many zero subdiagonal
//    entries.
func Dlaqr2(wantt, wantz bool, n, ktop, kbot, nw *int, h *mat.Matrix, ldh, iloz, ihiz *int, z *mat.Matrix, ldz, ns, nd *int, sr, si *mat.Vector, v *mat.Matrix, ldv, nh *int, t *mat.Matrix, ldt, nv *int, wv *mat.Matrix, ldwv *int, work *mat.Vector, lwork *int) {
	var bulge, sorted bool
	var aa, bb, beta, cc, cs, dd, evi, evk, foo, one, s, safmax, safmin, smlnum, sn, tau, ulp, zero float64
	var i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, kln, krow, kwtop, ltop, lwk1, lwk2, lwkopt int

	zero = 0.0
	one = 1.0

	//     ==== Estimate optimal workspace. ====
	jw = minint(*nw, (*kbot)-(*ktop)+1)
	if jw <= 2 {
		lwkopt = 1
	} else {
		//        ==== Workspace query call to DGEHRD ====
		Dgehrd(&jw, func() *int { y := 1; return &y }(), toPtr(jw-1), t, ldt, work, work, toPtr(-1), &info)
		lwk1 = int(work.Get(0))

		//        ==== Workspace query call to DORMHR ====
		Dormhr('R', 'N', &jw, &jw, func() *int { y := 1; return &y }(), toPtr(jw-1), t, ldt, work, v, ldv, work, toPtr(-1), &info)
		lwk2 = int(work.Get(0))

		//        ==== Optimal workspace ====
		lwkopt = jw + maxint(lwk1, lwk2)
	}

	//     ==== Quick return in case of workspace query. ====
	if (*lwork) == -1 {
		work.Set(0, float64(lwkopt))
		return
	}

	//     ==== Nothing to do ...
	//     ... for an empty active block ... ====
	(*ns) = 0
	(*nd) = 0
	work.Set(0, one)
	if (*ktop) > (*kbot) {
		return
	}
	//     ... nor for an empty deflation window. ====
	if (*nw) < 1 {
		return
	}

	//     ==== Machine constants ====
	safmin = Dlamch(SafeMinimum)
	safmax = one / safmin
	Dlabad(&safmin, &safmax)
	ulp = Dlamch(Precision)
	smlnum = safmin * (float64(*n) / ulp)

	//     ==== Setup deflation window ====
	jw = minint(*nw, (*kbot)-(*ktop)+1)
	kwtop = (*kbot) - jw + 1
	if kwtop == (*ktop) {
		s = zero
	} else {
		s = h.Get(kwtop-1, kwtop-1-1)
	}

	if (*kbot) == kwtop {
		//        ==== 1-by-1 deflation window: not much to do ====
		sr.Set(kwtop-1, h.Get(kwtop-1, kwtop-1))
		si.Set(kwtop-1, zero)
		(*ns) = 1
		(*nd) = 0
		if math.Abs(s) <= maxf64(smlnum, ulp*math.Abs(h.Get(kwtop-1, kwtop-1))) {
			(*ns) = 0
			(*nd) = 1
			if kwtop > (*ktop) {
				h.Set(kwtop-1, kwtop-1-1, zero)
			}
		}
		work.Set(0, one)
		return
	}

	//     ==== Convert to spike-triangular form.  (In case of a
	//     .    rare QR failure, this routine continues to do
	//     .    aggressive early deflation using that part of
	//     .    the deflation window that converged using INFQR
	//     .    here and there to keep track.) ====
	Dlacpy('U', &jw, &jw, h.Off(kwtop-1, kwtop-1), ldh, t, ldt)
	goblas.Dcopy(toPtr(jw-1), h.Vector(kwtop+1-1, kwtop-1), toPtr((*ldh)+1), t.Vector(1, 0), toPtr((*ldt)+1))

	Dlaset('A', &jw, &jw, &zero, &one, v, ldv)
	Dlahqr(true, true, &jw, func() *int { y := 1; return &y }(), &jw, t, ldt, sr.Off(kwtop-1), si.Off(kwtop-1), func() *int { y := 1; return &y }(), &jw, v, ldv, &infqr)

	//     ==== DTREXC needs a clean margin near the diagonal ====
	for j = 1; j <= jw-3; j++ {
		t.Set(j+2-1, j-1, zero)
		t.Set(j+3-1, j-1, zero)
	}
	if jw > 2 {
		t.Set(jw-1, jw-2-1, zero)
	}

	//     ==== Deflation detection loop ====
	(*ns) = jw
	ilst = infqr + 1
label20:
	;
	if ilst <= (*ns) {
		if (*ns) == 1 {
			bulge = false
		} else {
			bulge = t.Get((*ns)-1, (*ns)-1-1) != zero
		}

		//        ==== Small spike tip test for deflation ====
		if !bulge {
			//           ==== Real eigenvalue ====
			foo = math.Abs(t.Get((*ns)-1, (*ns)-1))
			if foo == zero {
				foo = math.Abs(s)
			}
			if math.Abs(s*v.Get(0, (*ns)-1)) <= maxf64(smlnum, ulp*foo) {
				//              ==== Deflatable ====
				(*ns) = (*ns) - 1
			} else {
				//              ==== Undeflatable.   Move it up out of the way.
				//              .    (DTREXC can not fail in this case.) ====
				ifst = (*ns)
				Dtrexc('V', &jw, t, ldt, v, ldv, &ifst, &ilst, work, &info)
				ilst = ilst + 1
			}
		} else {
			//           ==== Complex conjugate pair ====
			foo = math.Abs(t.Get((*ns)-1, (*ns)-1)) + math.Sqrt(math.Abs(t.Get((*ns)-1, (*ns)-1-1)))*math.Sqrt(math.Abs(t.Get((*ns)-1-1, (*ns)-1)))
			if foo == zero {
				foo = math.Abs(s)
			}
			if maxf64(math.Abs(s*v.Get(0, (*ns)-1)), math.Abs(s*v.Get(0, (*ns)-1-1))) <= maxf64(smlnum, ulp*foo) {
				//              ==== Deflatable ====
				(*ns) = (*ns) - 2
			} else {
				//              ==== Undeflatable. Move them up out of the way.
				//              .    Fortunately, DTREXC does the right thing with
				//              .    ILST in case of a rare exchange failure. ====
				ifst = (*ns)
				Dtrexc('V', &jw, t, ldt, v, ldv, &ifst, &ilst, work, &info)
				ilst = ilst + 2
			}
		}

		//        ==== End deflation detection loop ====
		goto label20
	}

	//        ==== Return to Hessenberg form ====
	if (*ns) == 0 {
		s = zero
	}

	if (*ns) < jw {
		//        ==== sorting diagonal blocks of T improves accuracy for
		//        .    graded matrices.  Bubble sort deals well with
		//        .    exchange failures. ====
		sorted = false
		i = (*ns) + 1
	label30:
		;
		if sorted {
			goto label50
		}
		sorted = true

		kend = i - 1
		i = infqr + 1
		if i == (*ns) {
			k = i + 1
		} else if t.Get(i+1-1, i-1) == zero {
			k = i + 1
		} else {
			k = i + 2
		}
	label40:
		;
		if k <= kend {
			if k == i+1 {
				evi = math.Abs(t.Get(i-1, i-1))
			} else {
				evi = math.Abs(t.Get(i-1, i-1)) + math.Sqrt(math.Abs(t.Get(i+1-1, i-1)))*math.Sqrt(math.Abs(t.Get(i-1, i+1-1)))
			}

			if k == kend {
				evk = math.Abs(t.Get(k-1, k-1))
			} else if t.Get(k+1-1, k-1) == zero {
				evk = math.Abs(t.Get(k-1, k-1))
			} else {
				evk = math.Abs(t.Get(k-1, k-1)) + math.Sqrt(math.Abs(t.Get(k+1-1, k-1)))*math.Sqrt(math.Abs(t.Get(k-1, k+1-1)))
			}

			if evi >= evk {
				i = k
			} else {
				sorted = false
				ifst = i
				ilst = k
				Dtrexc('V', &jw, t, ldt, v, ldv, &ifst, &ilst, work, &info)
				if info == 0 {
					i = ilst
				} else {
					i = k
				}
			}
			if i == kend {
				k = i + 1
			} else if t.Get(i+1-1, i-1) == zero {
				k = i + 1
			} else {
				k = i + 2
			}
			goto label40
		}
		goto label30
	label50:
	}

	//     ==== Restore shift/eigenvalue array from T ====
	i = jw
label60:
	;
	if i >= infqr+1 {
		if i == infqr+1 {
			sr.Set(kwtop+i-1-1, t.Get(i-1, i-1))
			si.Set(kwtop+i-1-1, zero)
			i = i - 1
		} else if t.Get(i-1, i-1-1) == zero {
			sr.Set(kwtop+i-1-1, t.Get(i-1, i-1))
			si.Set(kwtop+i-1-1, zero)
			i = i - 1
		} else {
			aa = t.Get(i-1-1, i-1-1)
			cc = t.Get(i-1, i-1-1)
			bb = t.Get(i-1-1, i-1)
			dd = t.Get(i-1, i-1)
			Dlanv2(&aa, &bb, &cc, &dd, sr.GetPtr(kwtop+i-2-1), si.GetPtr(kwtop+i-2-1), sr.GetPtr(kwtop+i-1-1), si.GetPtr(kwtop+i-1-1), &cs, &sn)
			i = i - 2
		}
		goto label60
	}

	if (*ns) < jw || s == zero {
		if (*ns) > 1 && s != zero {
			//           ==== Reflect spike back into lower triangle ====
			goblas.Dcopy(ns, v.VectorIdx(0), ldv, work, toPtr(1))
			beta = work.Get(0)
			Dlarfg(ns, &beta, work.Off(1), func() *int { y := 1; return &y }(), &tau)
			work.Set(0, one)

			Dlaset('L', toPtr(jw-2), toPtr(jw-2), &zero, &zero, t.Off(2, 0), ldt)

			Dlarf('L', ns, &jw, work, func() *int { y := 1; return &y }(), &tau, t, ldt, work.Off(jw+1-1))
			Dlarf('R', ns, ns, work, func() *int { y := 1; return &y }(), &tau, t, ldt, work.Off(jw+1-1))
			Dlarf('R', &jw, ns, work, func() *int { y := 1; return &y }(), &tau, v, ldv, work.Off(jw+1-1))

			Dgehrd(&jw, func() *int { y := 1; return &y }(), ns, t, ldt, work, work.Off(jw+1-1), toPtr((*lwork)-jw), &info)
		}

		//        ==== Copy updated reduced window into place ====
		if kwtop > 1 {
			h.Set(kwtop-1, kwtop-1-1, s*v.Get(0, 0))
		}
		Dlacpy('U', &jw, &jw, t, ldt, h.Off(kwtop-1, kwtop-1), ldh)
		goblas.Dcopy(toPtr(jw-1), t.Vector(1, 0), toPtr((*ldt)+1), h.Vector(kwtop+1-1, kwtop-1), toPtr((*ldh)+1))

		//        ==== Accumulate orthogonal matrix in order update
		//        .    H and Z, if requested.  ====
		if (*ns) > 1 && s != zero {
			Dormhr('R', 'N', &jw, ns, func() *int { y := 1; return &y }(), ns, t, ldt, work, v, ldv, work.Off(jw+1-1), toPtr((*lwork)-jw), &info)
		}

		//        ==== Update vertical slab in H ====
		if wantt {
			ltop = 1
		} else {
			ltop = (*ktop)
		}
		for krow = ltop; krow <= kwtop-1; krow += (*nv) {
			kln = minint(*nv, kwtop-krow)
			goblas.Dgemm(NoTrans, NoTrans, &kln, &jw, &jw, &one, h.Off(krow-1, kwtop-1), ldh, v, ldv, &zero, wv, ldwv)
			Dlacpy('A', &kln, &jw, wv, ldwv, h.Off(krow-1, kwtop-1), ldh)
		}

		//        ==== Update horizontal slab in H ====
		if wantt {
			for kcol = (*kbot) + 1; kcol <= (*n); kcol += (*nh) {
				kln = minint(*nh, (*n)-kcol+1)
				goblas.Dgemm(ConjTrans, NoTrans, &jw, &kln, &jw, &one, v, ldv, h.Off(kwtop-1, kcol-1), ldh, &zero, t, ldt)
				Dlacpy('A', &jw, &kln, t, ldt, h.Off(kwtop-1, kcol-1), ldh)
			}
		}

		//        ==== Update vertical slab in Z ====
		if wantz {
			for krow = (*iloz); krow <= (*ihiz); krow += (*nv) {
				kln = minint(*nv, (*ihiz)-krow+1)
				goblas.Dgemm(NoTrans, NoTrans, &kln, &jw, &jw, &one, z.Off(krow-1, kwtop-1), ldz, v, ldv, &zero, wv, ldwv)
				Dlacpy('A', &kln, &jw, wv, ldwv, z.Off(krow-1, kwtop-1), ldz)
			}
		}
	}

	//     ==== Return the number of deflations ... ====
	(*nd) = jw - (*ns)

	//     ==== ... and the number of shifts. (Subtracting
	//     .    INFQR from the spike length takes care
	//     .    of the case of a rare QR failure while
	//     .    calculating eigenvalues of the deflation
	//     .    window.)  ====
	(*ns) = (*ns) - infqr

	//      ==== Return optimal workspace. ====
	work.Set(0, float64(lwkopt))
}
