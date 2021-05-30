package golapack

import "golinalg/mat"

// Zhb2stkernels is an internal routine used by the ZHETRD_HB2ST
// subroutine.
func Zhb2stkernels(uplo byte, wantz bool, ttype, st, ed, sweep, n, nb, ib *int, a *mat.CMatrix, lda *int, v, tau *mat.CVector, ldvt *int, work *mat.CVector) {
	var upper bool
	var ctmp, one, zero complex128
	var dpos, i, j1, j2, lm, ln, ofdpos, taupos, vpos int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)

	// ajeter = (*ib) + (*ldvt)
	upper = uplo == 'U'
	if upper {
		dpos = 2*(*nb) + 1
		ofdpos = 2 * (*nb)
	} else {
		dpos = 1
		ofdpos = 2
	}

	//     Upper case
	if upper {

		if wantz {
			vpos = (((*sweep)-1)%2)*(*n) + (*st)
			taupos = (((*sweep)-1)%2)*(*n) + (*st)
		} else {
			vpos = (((*sweep)-1)%2)*(*n) + (*st)
			taupos = (((*sweep)-1)%2)*(*n) + (*st)
		}

		if (*ttype) == 1 {
			lm = (*ed) - (*st) + 1

			v.Set(vpos-1, one)
			for i = 1; i <= lm-1; i++ {
				v.Set(vpos+i-1, a.GetConj(ofdpos-i-1, (*st)+i-1))
				a.Set(ofdpos-i-1, (*st)+i-1, zero)
			}
			ctmp = a.GetConj(ofdpos-1, (*st)-1)
			Zlarfg(&lm, &ctmp, v.Off(vpos+1-1), func() *int { y := 1; return &y }(), tau.GetPtr(taupos-1))
			a.Set(ofdpos-1, (*st)-1, ctmp)

			lm = (*ed) - (*st) + 1
			Zlarfy(uplo, &lm, v.Off(vpos-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(taupos-1)), a.Off(dpos-1, (*st)-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)
		}

		if (*ttype) == 3 {

			lm = (*ed) - (*st) + 1
			Zlarfy(uplo, &lm, v.Off(vpos-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(taupos-1)), a.Off(dpos-1, (*st)-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)
		}

		if (*ttype) == 2 {
			j1 = (*ed) + 1
			j2 = minint((*ed)+(*nb), *n)
			ln = (*ed) - (*st) + 1
			lm = j2 - j1 + 1
			if lm > 0 {
				Zlarfx('L', &ln, &lm, v.Off(vpos-1), toPtrc128(tau.GetConj(taupos-1)), a.Off(dpos-(*nb)-1, j1-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)

				if wantz {
					vpos = (((*sweep)-1)%2)*(*n) + j1
					taupos = (((*sweep)-1)%2)*(*n) + j1
				} else {
					vpos = (((*sweep)-1)%2)*(*n) + j1
					taupos = (((*sweep)-1)%2)*(*n) + j1
				}

				v.Set(vpos-1, one)
				for i = 1; i <= lm-1; i++ {
					v.Set(vpos+i-1, a.GetConj(dpos-(*nb)-i-1, j1+i-1))
					a.Set(dpos-(*nb)-i-1, j1+i-1, zero)
				}
				ctmp = a.GetConj(dpos-(*nb)-1, j1-1)
				Zlarfg(&lm, &ctmp, v.Off(vpos+1-1), func() *int { y := 1; return &y }(), tau.GetPtr(taupos-1))
				a.Set(dpos-(*nb)-1, j1-1, ctmp)

				Zlarfx('R', toPtr(ln-1), &lm, v.Off(vpos-1), tau.GetPtr(taupos-1), a.Off(dpos-(*nb)+1-1, j1-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)
			}
		}

		//     Lower case
	} else {

		if wantz {
			vpos = (((*sweep)-1)%2)*(*n) + (*st)
			taupos = (((*sweep)-1)%2)*(*n) + (*st)
		} else {
			vpos = (((*sweep)-1)%2)*(*n) + (*st)
			taupos = (((*sweep)-1)%2)*(*n) + (*st)
		}

		if (*ttype) == 1 {
			lm = (*ed) - (*st) + 1

			v.Set(vpos-1, one)
			for i = 1; i <= lm-1; i++ {
				v.Set(vpos+i-1, a.Get(ofdpos+i-1, (*st)-1-1))
				a.Set(ofdpos+i-1, (*st)-1-1, zero)
			}
			Zlarfg(&lm, a.GetPtr(ofdpos-1, (*st)-1-1), v.Off(vpos+1-1), func() *int { y := 1; return &y }(), tau.GetPtr(taupos-1))

			lm = (*ed) - (*st) + 1

			Zlarfy(uplo, &lm, v.Off(vpos-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(taupos-1)), a.Off(dpos-1, (*st)-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)
		}

		if (*ttype) == 3 {
			lm = (*ed) - (*st) + 1

			Zlarfy(uplo, &lm, v.Off(vpos-1), func() *int { y := 1; return &y }(), toPtrc128(tau.GetConj(taupos-1)), a.Off(dpos-1, (*st)-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)
		}

		if (*ttype) == 2 {
			j1 = (*ed) + 1
			j2 = minint((*ed)+(*nb), *n)
			ln = (*ed) - (*st) + 1
			lm = j2 - j1 + 1

			if lm > 0 {
				Zlarfx('R', &lm, &ln, v.Off(vpos-1), tau.GetPtr(taupos-1), a.Off(dpos+(*nb)-1, (*st)-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)

				if wantz {
					vpos = (((*sweep)-1)%2)*(*n) + j1
					taupos = (((*sweep)-1)%2)*(*n) + j1
				} else {
					vpos = (((*sweep)-1)%2)*(*n) + j1
					taupos = (((*sweep)-1)%2)*(*n) + j1
				}

				v.Set(vpos-1, one)
				for i = 1; i <= lm-1; i++ {
					v.Set(vpos+i-1, a.Get(dpos+(*nb)+i-1, (*st)-1))
					a.Set(dpos+(*nb)+i-1, (*st)-1, zero)
				}
				Zlarfg(&lm, a.GetPtr(dpos+(*nb)-1, (*st)-1), v.Off(vpos+1-1), func() *int { y := 1; return &y }(), tau.GetPtr(taupos-1))

				Zlarfx('L', &lm, toPtr(ln-1), v.Off(vpos-1), toPtrc128(tau.GetConj(taupos-1)), a.Off(dpos+(*nb)-1-1, (*st)+1-1).UpdateRows((*lda)-1), toPtr((*lda)-1), work)
			}
		}
	}
}
