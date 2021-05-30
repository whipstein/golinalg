package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zunbdb4 simultaneously bidiagonalizes the blocks of a tall and skinny
// matrix X with orthonomal columns:
//
//                            [ B11 ]
//      [ X11 ]   [ P1 |    ] [  0  ]
//      [-----] = [---------] [-----] Q1**T .
//      [ X21 ]   [    | P2 ] [ B21 ]
//                            [  0  ]
//
// X11 is P-by-Q, and X21 is (M-P)-by-Q. M-Q must be no larger than P,
// M-P, or Q. Routines ZUNBDB1, ZUNBDB2, and ZUNBDB3 handle cases in
// which M-Q is not the minimum dimension.
//
// The unitary matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
// and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
// Householder vectors.
//
// B11 and B12 are (M-Q)-by-(M-Q) bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Zunbdb4(m, p, q *int, x11 *mat.CMatrix, ldx11 *int, x21 *mat.CMatrix, ldx21 *int, theta, phi *mat.Vector, taup1, taup2, tauq1, phantom, work *mat.CVector, lwork, info *int) {
	var lquery bool
	var negone, one, zero complex128
	var c, s float64
	var childinfo, i, ilarf, iorbdb5, j, llarf, lorbdb5, lworkmin, lworkopt int

	negone = (-1.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test input arguments
	(*info) = 0
	lquery = (*lwork) == -1

	if (*m) < 0 {
		(*info) = -1
	} else if (*p) < (*m)-(*q) || (*m)-(*p) < (*m)-(*q) {
		(*info) = -2
	} else if (*q) < (*m)-(*q) || (*q) > (*m) {
		(*info) = -3
	} else if (*ldx11) < maxint(1, *p) {
		(*info) = -5
	} else if (*ldx21) < maxint(1, (*m)-(*p)) {
		(*info) = -7
	}

	//     Compute workspace
	if (*info) == 0 {
		ilarf = 2
		llarf = maxint((*q)-1, (*p)-1, (*m)-(*p)-1)
		iorbdb5 = 2
		lorbdb5 = (*q)
		lworkopt = ilarf + llarf - 1
		lworkopt = maxint(lworkopt, iorbdb5+lorbdb5-1)
		lworkmin = lworkopt
		work.SetRe(0, float64(lworkopt))
		if (*lwork) < lworkmin && !lquery {
			(*info) = -14
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNBDB4"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Reduce columns 1, ..., M-Q of X11 and X21
	for i = 1; i <= (*m)-(*q); i++ {

		if i == 1 {
			for j = 1; j <= (*m); j++ {
				phantom.Set(j-1, zero)
			}
			Zunbdb5(p, toPtr((*m)-(*p)), q, phantom.Off(0), func() *int { y := 1; return &y }(), phantom.Off((*p)+1-1), func() *int { y := 1; return &y }(), x11, ldx11, x21, ldx21, work.Off(iorbdb5-1), &lorbdb5, &childinfo)
			goblas.Zscal(p, &negone, phantom.Off(0), func() *int { y := 1; return &y }())
			Zlarfgp(p, phantom.GetPtr(0), phantom.Off(1), func() *int { y := 1; return &y }(), taup1.GetPtr(0))
			Zlarfgp(toPtr((*m)-(*p)), phantom.GetPtr((*p)+1-1), phantom.Off((*p)+2-1), func() *int { y := 1; return &y }(), taup2.GetPtr(0))
			theta.Set(i-1, math.Atan2(phantom.GetRe(0), phantom.GetRe((*p)+1-1)))
			c = math.Cos(theta.Get(i - 1))
			s = math.Sin(theta.Get(i - 1))
			phantom.Set(0, one)
			phantom.Set((*p)+1-1, one)
			Zlarf('L', p, q, phantom.Off(0), func() *int { y := 1; return &y }(), toPtrc128(taup1.GetConj(0)), x11, ldx11, work.Off(ilarf-1))
			Zlarf('L', toPtr((*m)-(*p)), q, phantom.Off((*p)+1-1), func() *int { y := 1; return &y }(), toPtrc128(taup2.GetConj(0)), x21, ldx21, work.Off(ilarf-1))
		} else {
			Zunbdb5(toPtr((*p)-i+1), toPtr((*m)-(*p)-i+1), toPtr((*q)-i+1), x11.CVector(i-1, i-1-1), func() *int { y := 1; return &y }(), x21.CVector(i-1, i-1-1), func() *int { y := 1; return &y }(), x11.Off(i-1, i-1), ldx11, x21.Off(i-1, i-1), ldx21, work.Off(iorbdb5-1), &lorbdb5, &childinfo)
			goblas.Zscal(toPtr((*p)-i+1), &negone, x11.CVector(i-1, i-1-1), func() *int { y := 1; return &y }())
			Zlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1-1), x11.CVector(i+1-1, i-1-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			Zlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1-1), x21.CVector(i+1-1, i-1-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
			theta.Set(i-1, math.Atan2(x11.GetRe(i-1, i-1-1), x21.GetRe(i-1, i-1-1)))
			c = math.Cos(theta.Get(i - 1))
			s = math.Sin(theta.Get(i - 1))
			x11.Set(i-1, i-1-1, one)
			x21.Set(i-1, i-1-1, one)
			Zlarf('L', toPtr((*p)-i+1), toPtr((*q)-i+1), x11.CVector(i-1, i-1-1), func() *int { y := 1; return &y }(), toPtrc128(taup1.GetConj(i-1)), x11.Off(i-1, i-1), ldx11, work.Off(ilarf-1))
			Zlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*q)-i+1), x21.CVector(i-1, i-1-1), func() *int { y := 1; return &y }(), toPtrc128(taup2.GetConj(i-1)), x21.Off(i-1, i-1), ldx21, work.Off(ilarf-1))
		}

		goblas.Zdrot(toPtr((*q)-i+1), x11.CVector(i-1, i-1), ldx11, x21.CVector(i-1, i-1), ldx21, &s, toPtrf64(-c))
		Zlacgv(toPtr((*q)-i+1), x21.CVector(i-1, i-1), ldx21)
		Zlarfgp(toPtr((*q)-i+1), x21.GetPtr(i-1, i-1), x21.CVector(i-1, i+1-1), ldx21, tauq1.GetPtr(i-1))
		c = x21.GetRe(i-1, i-1)
		x21.Set(i-1, i-1, one)
		Zlarf('R', toPtr((*p)-i), toPtr((*q)-i+1), x21.CVector(i-1, i-1), ldx21, tauq1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work.Off(ilarf-1))
		Zlarf('R', toPtr((*m)-(*p)-i), toPtr((*q)-i+1), x21.CVector(i-1, i-1), ldx21, tauq1.GetPtr(i-1), x21.Off(i+1-1, i-1), ldx21, work.Off(ilarf-1))
		Zlacgv(toPtr((*q)-i+1), x21.CVector(i-1, i-1), ldx21)
		if i < (*m)-(*q) {
			s = math.Sqrt(math.Pow(goblas.Dznrm2(toPtr((*p)-i), x11.CVector(i+1-1, i-1), func() *int { y := 1; return &y }()), 2) + math.Pow(goblas.Dznrm2(toPtr((*m)-(*p)-i), x21.CVector(i+1-1, i-1), func() *int { y := 1; return &y }()), 2))
			phi.Set(i-1, math.Atan2(s, c))
		}

	}

	//     Reduce the bottom-right portion of X11 to [ I 0 ]
	for i = (*m) - (*q) + 1; i <= (*p); i++ {
		Zlacgv(toPtr((*q)-i+1), x11.CVector(i-1, i-1), ldx11)
		Zlarfgp(toPtr((*q)-i+1), x11.GetPtr(i-1, i-1), x11.CVector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1))
		x11.Set(i-1, i-1, one)
		Zlarf('R', toPtr((*p)-i), toPtr((*q)-i+1), x11.CVector(i-1, i-1), ldx11, tauq1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work.Off(ilarf-1))
		Zlarf('R', toPtr((*q)-(*p)), toPtr((*q)-i+1), x11.CVector(i-1, i-1), ldx11, tauq1.GetPtr(i-1), x21.Off((*m)-(*q)+1-1, i-1), ldx21, work.Off(ilarf-1))
		Zlacgv(toPtr((*q)-i+1), x11.CVector(i-1, i-1), ldx11)
	}

	//     Reduce the bottom-right portion of X21 to [ 0 I ]
	for i = (*p) + 1; i <= (*q); i++ {
		Zlacgv(toPtr((*q)-i+1), x21.CVector((*m)-(*q)+i-(*p)-1, i-1), ldx21)
		Zlarfgp(toPtr((*q)-i+1), x21.GetPtr((*m)-(*q)+i-(*p)-1, i-1), x21.CVector((*m)-(*q)+i-(*p)-1, i+1-1), ldx21, tauq1.GetPtr(i-1))
		x21.Set((*m)-(*q)+i-(*p)-1, i-1, one)
		Zlarf('R', toPtr((*q)-i), toPtr((*q)-i+1), x21.CVector((*m)-(*q)+i-(*p)-1, i-1), ldx21, tauq1.GetPtr(i-1), x21.Off((*m)-(*q)+i-(*p)+1-1, i-1), ldx21, work.Off(ilarf-1))
		Zlacgv(toPtr((*q)-i+1), x21.CVector((*m)-(*q)+i-(*p)-1, i-1), ldx21)
	}
}
