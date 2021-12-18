package golapack

import "github.com/whipstein/golinalg/mat"

// Dsb2stKernels is an internal routine used by the DSYTRD_SB2ST
// subroutine.
func Dsb2stKernels(uplo mat.MatUplo, wantz bool, ttype, st, ed, sweep, n, nb, ib int, a *mat.Matrix, v, tau *mat.Vector, work *mat.Vector) {
	var upper bool
	var ctmp, one, zero float64
	var dpos, i, j1, j2, lm, ln, ofdpos, taupos, vpos int

	zero = 0.0
	one = 1.0

	// ajeter = (*ib) + (*ldvt)
	upper = uplo == Upper
	if upper {
		dpos = 2*nb + 1
		ofdpos = 2 * nb
	} else {
		dpos = 1
		ofdpos = 2
	}

	//     Upper case
	if upper {

		if wantz {
			vpos = ((sweep-1)%2)*n + st
			taupos = ((sweep-1)%2)*n + st
		} else {
			vpos = ((sweep-1)%2)*n + st
			taupos = ((sweep-1)%2)*n + st
		}

		if ttype == 1 {
			lm = ed - st + 1

			v.Set(vpos-1, one)
			for i = 1; i <= lm-1; i++ {
				v.Set(vpos+i-1, (a.Get(ofdpos-i-1, st+i-1)))
				a.Set(ofdpos-i-1, st+i-1, zero)
			}
			ctmp = (a.Get(ofdpos-1, st-1))
			ctmp, *tau.GetPtr(taupos - 1) = Dlarfg(lm, ctmp, v.Off(vpos), 1)
			a.Set(ofdpos-1, st-1, ctmp)

			lm = ed - st + 1
			Dlarfy(uplo, lm, v.Off(vpos-1), 1, tau.Get(taupos-1), a.Off(dpos-1, st-1).UpdateRows(a.Rows-1), work)
		}

		if ttype == 3 {

			lm = ed - st + 1
			Dlarfy(uplo, lm, v.Off(vpos-1), 1, tau.Get(taupos-1), a.Off(dpos-1, st-1).UpdateRows(a.Rows-1), work)
		}

		if ttype == 2 {
			j1 = ed + 1
			j2 = min(ed+nb, n)
			ln = ed - st + 1
			lm = j2 - j1 + 1
			if lm > 0 {
				Dlarfx(Left, ln, lm, v.Off(vpos-1), tau.Get(taupos-1), a.Off(dpos-nb-1, j1-1).UpdateRows(a.Rows-1), work)

				if wantz {
					vpos = ((sweep-1)%2)*n + j1
					taupos = ((sweep-1)%2)*n + j1
				} else {
					vpos = ((sweep-1)%2)*n + j1
					taupos = ((sweep-1)%2)*n + j1
				}

				v.Set(vpos-1, one)
				for i = 1; i <= lm-1; i++ {
					v.Set(vpos+i-1, (a.Get(dpos-nb-i-1, j1+i-1)))
					a.Set(dpos-nb-i-1, j1+i-1, zero)
				}
				ctmp = (a.Get(dpos-nb-1, j1-1))
				ctmp, *tau.GetPtr(taupos - 1) = Dlarfg(lm, ctmp, v.Off(vpos), 1)
				a.Set(dpos-nb-1, j1-1, ctmp)

				Dlarfx(Right, ln-1, lm, v.Off(vpos-1), tau.Get(taupos-1), a.Off(dpos-nb, j1-1).UpdateRows(a.Rows-1), work)
			}
		}

		//     Lower case
	} else {

		if wantz {
			vpos = ((sweep-1)%2)*n + st
			taupos = ((sweep-1)%2)*n + st
		} else {
			vpos = ((sweep-1)%2)*n + st
			taupos = ((sweep-1)%2)*n + st
		}

		if ttype == 1 {
			lm = ed - st + 1

			v.Set(vpos-1, one)
			for i = 1; i <= lm-1; i++ {
				v.Set(vpos+i-1, a.Get(ofdpos+i-1, st-1-1))
				a.Set(ofdpos+i-1, st-1-1, zero)
			}
			*a.GetPtr(ofdpos-1, st-1-1), *tau.GetPtr(taupos - 1) = Dlarfg(lm, a.Get(ofdpos-1, st-1-1), v.Off(vpos), 1)

			lm = ed - st + 1

			Dlarfy(uplo, lm, v.Off(vpos-1), 1, tau.Get(taupos-1), a.Off(dpos-1, st-1).UpdateRows(a.Rows-1), work)
		}

		if ttype == 3 {
			lm = ed - st + 1

			Dlarfy(uplo, lm, v.Off(vpos-1), 1, tau.Get(taupos-1), a.Off(dpos-1, st-1).UpdateRows(a.Rows-1), work)
		}

		if ttype == 2 {
			j1 = ed + 1
			j2 = min(ed+nb, n)
			ln = ed - st + 1
			lm = j2 - j1 + 1

			if lm > 0 {
				Dlarfx(Right, lm, ln, v.Off(vpos-1), tau.Get(taupos-1), a.Off(dpos+nb-1, st-1).UpdateRows(a.Rows-1), work)

				if wantz {
					vpos = ((sweep-1)%2)*n + j1
					taupos = ((sweep-1)%2)*n + j1
				} else {
					vpos = ((sweep-1)%2)*n + j1
					taupos = ((sweep-1)%2)*n + j1
				}

				v.Set(vpos-1, one)
				for i = 1; i <= lm-1; i++ {
					v.Set(vpos+i-1, a.Get(dpos+nb+i-1, st-1))
					a.Set(dpos+nb+i-1, st-1, zero)
				}
				*a.GetPtr(dpos+nb-1, st-1), *tau.GetPtr(taupos - 1) = Dlarfg(lm, a.Get(dpos+nb-1, st-1), v.Off(vpos), 1)

				Dlarfx(Left, lm, ln-1, v.Off(vpos-1), tau.Get(taupos-1), a.Off(dpos+nb-1-1, st).UpdateRows(a.Rows-1), work)
			}
		}
	}
}
