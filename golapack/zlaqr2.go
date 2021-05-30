package golapack

import (
	"golinalg/goblas"
	"golinalg/mat"
	"math/cmplx"
)

// Zlaqr2 is identical to ZLAQR3 except that it avoids
//    recursion by calling ZLAHQR instead of ZLAQR4.
//
//    Aggressive early deflation:
//
//    ZLAQR2 accepts as input an upper Hessenberg matrix
//    H and performs an unitary similarity transformation
//    designed to detect and deflate fully converged eigenvalues from
//    a trailing principal submatrix.  On output H has been over-
//    written by a new Hessenberg matrix that is a perturbation of
//    an unitary similarity transformation of H.  It is to be
//    hoped that the final version of H has many zero subdiagonal
//    entries.
func Zlaqr2(wantt, wantz bool, n, ktop, kbot, nw *int, h *mat.CMatrix, ldh, iloz, ihiz *int, z *mat.CMatrix, ldz, ns, nd *int, sh *mat.CVector, v *mat.CMatrix, ldv, nh *int, t *mat.CMatrix, ldt, nv *int, wv *mat.CMatrix, ldwv *int, work *mat.CVector, lwork *int) {
	var beta, one, s, tau, zero complex128
	var foo, rone, rzero, safmax, safmin, smlnum, ulp float64
	var i, ifst, ilst, info, infqr, j, jw, kcol, kln, knt, krow, kwtop, ltop, lwk1, lwk2, lwkopt int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0

	//     ==== Estimate optimal workspace. ====
	jw = minint(*nw, (*kbot)-(*ktop)+1)
	if jw <= 2 {
		lwkopt = 1
	} else {
		//        ==== Workspace query call to ZGEHRD ====
		Zgehrd(&jw, func() *int { y := 1; return &y }(), toPtr(jw-1), t, ldt, work, work, toPtr(-1), &info)
		lwk1 = int(work.GetRe(0))

		//        ==== Workspace query call to ZUNMHR ====
		Zunmhr('R', 'N', &jw, &jw, func() *int { y := 1; return &y }(), toPtr(jw-1), t, ldt, work, v, ldv, work, toPtr(-1), &info)
		lwk2 = int(work.GetRe(0))

		//        ==== Optimal workspace ====
		lwkopt = jw + maxint(lwk1, lwk2)
	}

	//     ==== Quick return in case of workspace query. ====
	if (*lwork) == -1 {
		work.Set(0, complex(float64(lwkopt), 0))
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
	safmax = rone / safmin
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
		sh.Set(kwtop-1, h.Get(kwtop-1, kwtop-1))
		(*ns) = 1
		(*nd) = 0
		if cabs1(s) <= maxf64(smlnum, ulp*cabs1(h.Get(kwtop-1, kwtop-1))) {
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
	Zlacpy('U', &jw, &jw, h.Off(kwtop-1, kwtop-1), ldh, t, ldt)
	goblas.Zcopy(toPtr(jw-1), h.CVector(kwtop+1-1, kwtop-1), toPtr((*ldh)+1), t.CVector(1, 0), toPtr((*ldt)+1))

	Zlaset('A', &jw, &jw, &zero, &one, v, ldv)
	Zlahqr(true, true, &jw, func() *int { y := 1; return &y }(), &jw, t, ldt, sh.Off(kwtop-1), func() *int { y := 1; return &y }(), &jw, v, ldv, &infqr)

	//     ==== Deflation detection loop ====
	(*ns) = jw
	ilst = infqr + 1
	for knt = infqr + 1; knt <= jw; knt++ {
		//        ==== Small spike tip deflation test ====
		foo = cabs1(t.Get((*ns)-1, (*ns)-1))
		if foo == rzero {
			foo = cabs1(s)
		}
		if cabs1(s)*cabs1(v.Get(0, (*ns)-1)) <= maxf64(smlnum, ulp*foo) {
			//           ==== One more converged eigenvalue ====
			(*ns) = (*ns) - 1
		} else {
			//           ==== One undeflatable eigenvalue.  Move it up out of the
			//           .    way.   (ZTREXC can not fail in this case.) ====
			ifst = (*ns)
			Ztrexc('V', &jw, t, ldt, v, ldv, &ifst, &ilst, &info)
			ilst = ilst + 1
		}
	}

	//        ==== Return to Hessenberg form ====
	if (*ns) == 0 {
		s = zero
	}

	if (*ns) < jw {
		//        ==== sorting the diagonal of T improves accuracy for
		//        .    graded matrices.  ====
		for i = infqr + 1; i <= (*ns); i++ {
			ifst = i
			for j = i + 1; j <= (*ns); j++ {
				if cabs1(t.Get(j-1, j-1)) > cabs1(t.Get(ifst-1, ifst-1)) {
					ifst = j
				}
			}
			ilst = i
			if ifst != ilst {
				Ztrexc('V', &jw, t, ldt, v, ldv, &ifst, &ilst, &info)
			}
		}
	}

	//     ==== Restore shift/eigenvalue array from T ====
	for i = infqr + 1; i <= jw; i++ {
		sh.Set(kwtop+i-1-1, t.Get(i-1, i-1))
	}

	if (*ns) < jw || s == zero {
		if (*ns) > 1 && s != zero {
			//           ==== Reflect spike back into lower triangle ====
			goblas.Zcopy(ns, v.CVector(0, 0), ldv, work, func() *int { y := 1; return &y }())
			for i = 1; i <= (*ns); i++ {
				work.Set(i-1, work.GetConj(i-1))
			}
			beta = work.Get(0)
			Zlarfg(ns, &beta, work.Off(1), func() *int { y := 1; return &y }(), &tau)
			work.Set(0, one)

			Zlaset('L', toPtr(jw-2), toPtr(jw-2), &zero, &zero, t.Off(2, 0), ldt)

			Zlarf('L', ns, &jw, work, func() *int { y := 1; return &y }(), toPtrc128(cmplx.Conj(tau)), t, ldt, work.Off(jw+1-1))
			Zlarf('R', ns, ns, work, func() *int { y := 1; return &y }(), &tau, t, ldt, work.Off(jw+1-1))
			Zlarf('R', &jw, ns, work, func() *int { y := 1; return &y }(), &tau, v, ldv, work.Off(jw+1-1))

			Zgehrd(&jw, func() *int { y := 1; return &y }(), ns, t, ldt, work, work.Off(jw+1-1), toPtr((*lwork)-jw), &info)
		}

		//        ==== Copy updated reduced window into place ====
		if kwtop > 1 {
			h.Set(kwtop-1, kwtop-1-1, s*v.GetConj(0, 0))
		}
		Zlacpy('U', &jw, &jw, t, ldt, h.Off(kwtop-1, kwtop-1), ldh)
		goblas.Zcopy(toPtr(jw-1), t.CVector(1, 0), toPtr((*ldt)+1), h.CVector(kwtop+1-1, kwtop-1), toPtr((*ldh)+1))

		//        ==== Accumulate orthogonal matrix in order update
		//        .    H and Z, if requested.  ====
		if (*ns) > 1 && s != zero {
			Zunmhr('R', 'N', &jw, ns, func() *int { y := 1; return &y }(), ns, t, ldt, work, v, ldv, work.Off(jw+1-1), toPtr((*lwork)-jw), &info)
		}

		//        ==== Update vertical slab in H ====
		if wantt {
			ltop = 1
		} else {
			ltop = (*ktop)
		}
		for krow = ltop; krow <= kwtop-1; krow += (*nv) {
			kln = minint(*nv, kwtop-krow)
			goblas.Zgemm('N', 'N', &kln, &jw, &jw, &one, h.Off(krow-1, kwtop-1), ldh, v, ldv, &zero, wv, ldwv)
			Zlacpy('A', &kln, &jw, wv, ldwv, h.Off(krow-1, kwtop-1), ldh)
		}

		//        ==== Update horizontal slab in H ====
		if wantt {
			for kcol = (*kbot) + 1; kcol <= (*n); kcol += (*nh) {
				kln = minint(*nh, (*n)-kcol+1)
				goblas.Zgemm('C', 'N', &jw, &kln, &jw, &one, v, ldv, h.Off(kwtop-1, kcol-1), ldh, &zero, t, ldt)
				Zlacpy('A', &jw, &kln, t, ldt, h.Off(kwtop-1, kcol-1), ldh)
			}
		}

		//        ==== Update vertical slab in Z ====
		if wantz {
			for krow = (*iloz); krow <= (*ihiz); krow += (*nv) {
				kln = minint(*nv, (*ihiz)-krow+1)
				goblas.Zgemm('N', 'N', &kln, &jw, &jw, &one, z.Off(krow-1, kwtop-1), ldz, v, ldv, &zero, wv, ldwv)
				Zlacpy('A', &kln, &jw, wv, ldwv, z.Off(krow-1, kwtop-1), ldz)
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
	work.Set(0, complex(float64(lwkopt), 0))
}
