package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/mat"
)

// Dlaqr3 accepts as input an upper Hessenberg matrix
//    H and performs an orthogonal similarity transformation
//    designed to detect and deflate fully converged eigenvalues from
//    a trailing principal submatrix.  On output H has been over-
//    written by a new Hessenberg matrix that is a perturbation of
//    an orthogonal similarity transformation of H.  It is to be
//    hoped that the final version of H has many zero subdiagonal
//    entries.
func Dlaqr3(wantt, wantz bool, n, ktop, kbot, nw int, h *mat.Matrix, iloz, ihiz int, z *mat.Matrix, sr, si *mat.Vector, v *mat.Matrix, nh int, t *mat.Matrix, nv int, wv *mat.Matrix, work *mat.Vector, lwork int) (ns, nd int) {
	var bulge, sorted bool
	var aa, bb, beta, cc, dd, evi, evk, foo, one, s, safmax, safmin, smlnum, tau, ulp, zero float64
	var i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, kln, krow, kwtop, ltop, lwk1, lwk2, lwk3, lwkopt, nmin int
	var err error

	zero = 0.0
	one = 1.0

	//     ==== Estimate optimal workspace. ====
	jw = min(nw, kbot-ktop+1)
	if jw <= 2 {
		lwkopt = 1
	} else {
		//        ==== Workspace query call to DGEHRD ====
		if err = Dgehrd(jw, 1, jw-1, t, work, work, -1); err != nil {
			panic(err)
		}
		lwk1 = int(work.Get(0))

		//        ==== Workspace query call to DORMHR ====
		if err = Dormhr(Right, NoTrans, jw, jw, 1, jw-1, t, work, v, work, -1); err != nil {
			panic(err)
		}
		lwk2 = int(work.Get(0))

		//        ==== Workspace query call to DLAQR4 ====
		infqr = Dlaqr4(true, true, jw, 1, jw, t, sr, si, 1, jw, v, work, -1)
		lwk3 = int(work.Get(0))

		//        ==== Optimal workspace ====
		lwkopt = max(jw+max(lwk1, lwk2), lwk3)
	}

	//     ==== Quick return in case of workspace query. ====
	if lwork == -1 {
		work.Set(0, float64(lwkopt))
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
	safmax = one / safmin
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
		sr.Set(kwtop-1, h.Get(kwtop-1, kwtop-1))
		si.Set(kwtop-1, zero)
		ns = 1
		nd = 0
		if math.Abs(s) <= math.Max(smlnum, ulp*math.Abs(h.Get(kwtop-1, kwtop-1))) {
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
	Dlacpy(Upper, jw, jw, h.Off(kwtop-1, kwtop-1), t)
	goblas.Dcopy(jw-1, h.Vector(kwtop, kwtop-1, h.Rows+1), t.Vector(1, 0, t.Rows+1))
	//
	Dlaset(Full, jw, jw, zero, one, v)
	nmin = Ilaenv(12, "Dlaqr3", []byte("SV"), jw, 1, jw, lwork)
	if jw > nmin {
		infqr = Dlaqr4(true, true, jw, 1, jw, t, sr.Off(kwtop-1), si.Off(kwtop-1), 1, jw, v, work, lwork)
	} else {
		if infqr, err = Dlahqr(true, true, jw, 1, jw, t, sr.Off(kwtop-1), si.Off(kwtop-1), 1, jw, v); err != nil {
			panic(err)
		}
	}

	//     ==== DTREXC needs a clean margin near the diagonal ====
	for j = 1; j <= jw-3; j++ {
		t.Set(j+2-1, j-1, zero)
		t.Set(j+3-1, j-1, zero)
	}
	if jw > 2 {
		t.Set(jw-1, jw-2-1, zero)
	}

	//     ==== Deflation detection loop ====
	ns = jw
	ilst = infqr + 1
label20:
	;
	if ilst <= ns {
		if ns == 1 {
			bulge = false
		} else {
			bulge = t.Get(ns-1, ns-1-1) != zero
		}

		//        ==== Small spike tip test for deflation ====
		if !bulge {
			//           ==== Real eigenvalue ====
			foo = math.Abs(t.Get(ns-1, ns-1))
			if foo == zero {
				foo = math.Abs(s)
			}
			if math.Abs(s*v.Get(0, ns-1)) <= math.Max(smlnum, ulp*foo) {
				//              ==== Deflatable ====
				ns = ns - 1
			} else {
				//              ==== Undeflatable.   Move it up out of the way.
				//              .    (DTREXC can not fail in this case.) ====
				ifst = ns
				if ifst, ilst, info, err = Dtrexc('V', jw, t, v, ifst, ilst, work); err != nil {
					panic(err)
				}
				ilst = ilst + 1
			}
		} else {
			//           ==== Complex conjugate pair ====
			foo = math.Abs(t.Get(ns-1, ns-1)) + math.Sqrt(math.Abs(t.Get(ns-1, ns-1-1)))*math.Sqrt(math.Abs(t.Get(ns-1-1, ns-1)))
			if foo == zero {
				foo = math.Abs(s)
			}
			if math.Max(math.Abs(s*v.Get(0, ns-1)), math.Abs(s*v.Get(0, ns-1-1))) <= math.Max(smlnum, ulp*foo) {
				//              ==== Deflatable ====
				ns = ns - 2
			} else {
				//              ==== Undeflatable. Move them up out of the way.
				//              .    Fortunately, DTREXC does the right thing with
				//              .    ILST in case of a rare exchange failure. ====
				ifst = ns
				if ifst, ilst, info, err = Dtrexc('V', jw, t, v, ifst, ilst, work); err != nil {
					panic(err)
				}
				ilst = ilst + 2
			}
		}

		//        ==== End deflation detection loop ====
		goto label20
	}

	//        ==== Return to Hessenberg form ====
	if ns == 0 {
		s = zero
	}

	if ns < jw {
		//        ==== sorting diagonal blocks of T improves accuracy for
		//        .    graded matrices.  Bubble sort deals well with
		//        .    exchange failures. ====
		sorted = false
		i = ns + 1
	label30:
		;
		if sorted {
			goto label50
		}
		sorted = true

		kend = i - 1
		i = infqr + 1
		if i == ns {
			k = i + 1
		} else if t.Get(i, i-1) == zero {
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
				evi = math.Abs(t.Get(i-1, i-1)) + math.Sqrt(math.Abs(t.Get(i, i-1)))*math.Sqrt(math.Abs(t.Get(i-1, i)))
			}

			if k == kend {
				evk = math.Abs(t.Get(k-1, k-1))
			} else if t.Get(k, k-1) == zero {
				evk = math.Abs(t.Get(k-1, k-1))
			} else {
				evk = math.Abs(t.Get(k-1, k-1)) + math.Sqrt(math.Abs(t.Get(k, k-1)))*math.Sqrt(math.Abs(t.Get(k-1, k)))
			}

			if evi >= evk {
				i = k
			} else {
				sorted = false
				ifst = i
				ilst = k
				if ifst, ilst, info, err = Dtrexc('V', jw, t, v, ifst, ilst, work); err != nil {
					panic(err)
				}
				if info == 0 {
					i = ilst
				} else {
					i = k
				}
			}
			if i == kend {
				k = i + 1
			} else if t.Get(i, i-1) == zero {
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
			aa, bb, cc, dd, *sr.GetPtr(kwtop + i - 2 - 1), *si.GetPtr(kwtop + i - 2 - 1), *sr.GetPtr(kwtop + i - 1 - 1), *si.GetPtr(kwtop + i - 1 - 1), _, _ = Dlanv2(aa, bb, cc, dd)
			i = i - 2
		}
		goto label60
	}

	if ns < jw || s == zero {
		if ns > 1 && s != zero {
			//           ==== Reflect spike back into lower triangle ====
			goblas.Dcopy(ns, v.VectorIdx(0), work)
			beta = work.Get(0)
			beta, tau = Dlarfg(ns, beta, work.Off(1, 1))
			work.Set(0, one)

			Dlaset(Lower, jw-2, jw-2, zero, zero, t.Off(2, 0))

			Dlarf(Left, ns, jw, work.Off(0, 1), tau, t, work.Off(jw))
			Dlarf(Right, ns, ns, work.Off(0, 1), tau, t, work.Off(jw))
			Dlarf(Right, jw, ns, work.Off(0, 1), tau, v, work.Off(jw))

			if err = Dgehrd(jw, 1, ns, t, work, work.Off(jw), lwork-jw); err != nil {
				panic(err)
			}
		}

		//        ==== Copy updated reduced window into place ====
		if kwtop > 1 {
			h.Set(kwtop-1, kwtop-1-1, s*v.Get(0, 0))
		}
		Dlacpy(Upper, jw, jw, t, h.Off(kwtop-1, kwtop-1))
		goblas.Dcopy(jw-1, t.Vector(1, 0, t.Rows+1), h.Vector(kwtop, kwtop-1, h.Rows+1))

		//        ==== Accumulate orthogonal matrix in order update
		//        .    H and Z, if requested.  ====
		if ns > 1 && s != zero {
			if err = Dormhr(Right, NoTrans, jw, ns, 1, ns, t, work, v, work.Off(jw), lwork-jw); err != nil {
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
			err = goblas.Dgemm(NoTrans, NoTrans, kln, jw, jw, one, h.Off(krow-1, kwtop-1), v, zero, wv)
			Dlacpy(Full, kln, jw, wv, h.Off(krow-1, kwtop-1))
		}

		//        ==== Update horizontal slab in H ====
		if wantt {
			for kcol = kbot + 1; kcol <= n; kcol += nh {
				kln = min(nh, n-kcol+1)
				err = goblas.Dgemm(ConjTrans, NoTrans, jw, kln, jw, one, v, h.Off(kwtop-1, kcol-1), zero, t)
				Dlacpy(Full, jw, kln, t, h.Off(kwtop-1, kcol-1))
			}
		}

		//        ==== Update vertical slab in Z ====
		if wantz {
			for krow = iloz; krow <= ihiz; krow += nv {
				kln = min(nv, ihiz-krow+1)
				err = goblas.Dgemm(NoTrans, NoTrans, kln, jw, jw, one, z.Off(krow-1, kwtop-1), v, zero, wv)
				Dlacpy(Full, kln, jw, wv, z.Off(krow-1, kwtop-1))
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
	work.Set(0, float64(lwkopt))

	return
}
