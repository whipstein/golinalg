package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
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
func Zlaqr2(wantt, wantz bool, n, ktop, kbot, nw int, h *mat.CMatrix, iloz, ihiz int, z *mat.CMatrix, sh *mat.CVector, v *mat.CMatrix, nh int, t *mat.CMatrix, nv int, wv *mat.CMatrix, work *mat.CVector, lwork int) (ns, nd int) {
	var beta, one, s, tau, zero complex128
	var foo, rone, rzero, safmax, safmin, smlnum, ulp float64
	var i, ifst, ilst, infqr, j, jw, kcol, kln, knt, krow, kwtop, ltop, lwk1, lwk2, lwkopt int
	var err error

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0

	//     ==== Estimate optimal workspace. ====
	jw = min(nw, kbot-ktop+1)
	if jw <= 2 {
		lwkopt = 1
	} else {
		//        ==== Workspace query call to ZGEHRD ====
		if err = Zgehrd(jw, 1, jw-1, t, work, work, -1); err != nil {
			panic(err)
		}
		lwk1 = int(work.GetRe(0))

		//        ==== Workspace query call to ZUNMHR ====
		if err = Zunmhr(Right, NoTrans, jw, jw, 1, jw-1, t, work, v, work, -1); err != nil {
			panic(err)
		}
		lwk2 = int(work.GetRe(0))

		//        ==== Optimal workspace ====
		lwkopt = jw + max(lwk1, lwk2)
	}

	//     ==== Quick return in case of workspace query. ====
	if lwork == -1 {
		work.Set(0, complex(float64(lwkopt), 0))
		return
	}

	//     ==== Nothing to do ...
	//     ... for an empty active block ... ====
	ns = 0
	nd = 0
	work.Set(0, one)
	if ktop > kbot {
		return
	}
	//     ... nor for an empty deflation window. ====
	if nw < 1 {
		return
	}

	//     ==== Machine constants ====
	safmin = Dlamch(SafeMinimum)
	safmax = rone / safmin
	safmin, safmax = Dlabad(safmin, safmax)
	ulp = Dlamch(Precision)
	smlnum = safmin * (float64(n) / ulp)

	//     ==== Setup deflation window ====
	jw = min(nw, kbot-ktop+1)
	kwtop = kbot - jw + 1
	if kwtop == ktop {
		s = zero
	} else {
		s = h.Get(kwtop-1, kwtop-1-1)
	}

	if kbot == kwtop {
		//        ==== 1-by-1 deflation window: not much to do ====
		sh.Set(kwtop-1, h.Get(kwtop-1, kwtop-1))
		ns = 1
		nd = 0
		if cabs1(s) <= math.Max(smlnum, ulp*cabs1(h.Get(kwtop-1, kwtop-1))) {
			ns = 0
			nd = 1
			if kwtop > ktop {
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
	Zlacpy(Upper, jw, jw, h.Off(kwtop-1, kwtop-1), t)
	goblas.Zcopy(jw-1, h.CVector(kwtop, kwtop-1, h.Rows+1), t.CVector(1, 0, (*&t.Rows)+1))

	Zlaset(Full, jw, jw, zero, one, v)
	infqr = Zlahqr(true, true, jw, 1, jw, t, sh.Off(kwtop-1), 1, jw, v)

	//     ==== Deflation detection loop ====
	ns = jw
	ilst = infqr + 1
	for knt = infqr + 1; knt <= jw; knt++ {
		//        ==== Small spike tip deflation test ====
		foo = cabs1(t.Get(ns-1, ns-1))
		if foo == rzero {
			foo = cabs1(s)
		}
		if cabs1(s)*cabs1(v.Get(0, ns-1)) <= math.Max(smlnum, ulp*foo) {
			//           ==== One more converged eigenvalue ====
			ns = ns - 1
		} else {
			//           ==== One undeflatable eigenvalue.  Move it up out of the
			//           .    way.   (ZTREXC can not fail in this case.) ====
			ifst = ns
			if err = Ztrexc('V', jw, t, v, ifst, ilst); err != nil {
				panic(err)
			}
			ilst = ilst + 1
		}
	}

	//        ==== Return to Hessenberg form ====
	if ns == 0 {
		s = zero
	}

	if ns < jw {
		//        ==== sorting the diagonal of T improves accuracy for
		//        .    graded matrices.  ====
		for i = infqr + 1; i <= ns; i++ {
			ifst = i
			for j = i + 1; j <= ns; j++ {
				if cabs1(t.Get(j-1, j-1)) > cabs1(t.Get(ifst-1, ifst-1)) {
					ifst = j
				}
			}
			ilst = i
			if ifst != ilst {
				if err = Ztrexc('V', jw, t, v, ifst, ilst); err != nil {
					panic(err)
				}
			}
		}
	}

	//     ==== Restore shift/eigenvalue array from T ====
	for i = infqr + 1; i <= jw; i++ {
		sh.Set(kwtop+i-1-1, t.Get(i-1, i-1))
	}

	if ns < jw || s == zero {
		if ns > 1 && s != zero {
			//           ==== Reflect spike back into lower triangle ====
			goblas.Zcopy(ns, v.CVector(0, 0, *&v.Rows), work.Off(0, 1))
			for i = 1; i <= ns; i++ {
				work.Set(i-1, work.GetConj(i-1))
			}
			beta = work.Get(0)
			beta, tau = Zlarfg(ns, beta, work.Off(1, 1))
			work.Set(0, one)

			Zlaset(Lower, jw-2, jw-2, zero, zero, t.Off(2, 0))

			Zlarf(Left, ns, jw, work.Off(0, 1), cmplx.Conj(tau), t, work.Off(jw))
			Zlarf(Right, ns, ns, work.Off(0, 1), tau, t, work.Off(jw))
			Zlarf(Right, jw, ns, work.Off(0, 1), tau, v, work.Off(jw))

			if err = Zgehrd(jw, 1, ns, t, work, work.Off(jw), lwork-jw); err != nil {
				panic(err)
			}
		}

		//        ==== Copy updated reduced window into place ====
		if kwtop > 1 {
			h.Set(kwtop-1, kwtop-1-1, s*v.GetConj(0, 0))
		}
		Zlacpy(Upper, jw, jw, t, h.Off(kwtop-1, kwtop-1))
		goblas.Zcopy(jw-1, t.CVector(1, 0, (*&t.Rows)+1), h.CVector(kwtop, kwtop-1, h.Rows+1))

		//        ==== Accumulate orthogonal matrix in order update
		//        .    H and Z, if requested.  ====
		if ns > 1 && s != zero {
			if err = Zunmhr(Right, NoTrans, jw, ns, 1, ns, t, work, v, work.Off(jw), lwork-jw); err != nil {
				panic(err)
			}
		}

		//        ==== Update vertical slab in H ====
		if wantt {
			ltop = 1
		} else {
			ltop = ktop
		}
		for krow = ltop; krow <= kwtop-1; krow += nv {
			kln = min(nv, kwtop-krow)
			if err = goblas.Zgemm(NoTrans, NoTrans, kln, jw, jw, one, h.Off(krow-1, kwtop-1), v, zero, wv); err != nil {
				panic(err)
			}
			Zlacpy(Full, kln, jw, wv, h.Off(krow-1, kwtop-1))
		}

		//        ==== Update horizontal slab in H ====
		if wantt {
			for kcol = kbot + 1; kcol <= n; kcol += nh {
				kln = min(nh, n-kcol+1)
				if err = goblas.Zgemm(ConjTrans, NoTrans, jw, kln, jw, one, v, h.Off(kwtop-1, kcol-1), zero, t); err != nil {
					panic(err)
				}
				Zlacpy(Full, jw, kln, t, h.Off(kwtop-1, kcol-1))
			}
		}

		//        ==== Update vertical slab in Z ====
		if wantz {
			for krow = iloz; krow <= ihiz; krow += nv {
				kln = min(nv, ihiz-krow+1)
				if err = goblas.Zgemm(NoTrans, NoTrans, kln, jw, jw, one, z.Off(krow-1, kwtop-1), v, zero, wv); err != nil {
					panic(err)
				}
				Zlacpy(Full, kln, jw, wv, z.Off(krow-1, kwtop-1))
			}
		}
	}

	//     ==== Return the number of deflations ... ====
	nd = jw - ns

	//     ==== ... and the number of shifts. (Subtracting
	//     .    INFQR from the spike length takes care
	//     .    of the case of a rare QR failure while
	//     .    calculating eigenvalues of the deflation
	//     .    window.)  ====
	ns = ns - infqr

	//      ==== Return optimal workspace. ====
	work.Set(0, complex(float64(lwkopt), 0))

	return
}
