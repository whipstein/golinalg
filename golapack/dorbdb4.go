package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dorbdb4 simultaneously bidiagonalizes the blocks of a tall and skinny
// matrix X with orthonomal columns:
//
//                            [ B11 ]
//      [ X11 ]   [ P1 |    ] [  0  ]
//      [-----] = [---------] [-----] Q1**T .
//      [ X21 ]   [    | P2 ] [ B21 ]
//                            [  0  ]
//
// X11 is P-by-Q, and X21 is (M-P)-by-Q. M-Q must be no larger than P,
// M-P, or Q. Routines DORBDB1, DORBDB2, and DORBDB3 handle cases in
// which M-Q is not the minimum dimension.
//
// The orthogonal matrices P1, P2, and Q1 are P-by-P, (M-P)-by-(M-P),
// and (M-Q)-by-(M-Q), respectively. They are represented implicitly by
// Householder vectors.
//
// B11 and B12 are (M-Q)-by-(M-Q) bidiagonal matrices represented
// implicitly by angles THETA, PHI.
func Dorbdb4(m, p, q *int, x11 *mat.Matrix, ldx11 *int, x21 *mat.Matrix, ldx21 *int, theta, phi, taup1, taup2, tauq1, phantom, work *mat.Vector, lwork, info *int) {
	var lquery bool
	var c, negone, one, s, zero float64
	var childinfo, i, ilarf, iorbdb5, j, llarf, lorbdb5, lworkmin, lworkopt int

	negone = -1.0
	one = 1.0
	zero = 0.0

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
		work.Set(0, float64(lworkopt))
		if (*lwork) < lworkmin && !lquery {
			(*info) = -14
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DORBDB4"), -(*info))
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
			Dorbdb5(p, toPtr((*m)-(*p)), q, phantom, func() *int { y := 1; return &y }(), phantom.Off((*p)+1-1), func() *int { y := 1; return &y }(), x11, ldx11, x21, ldx21, work.Off(iorbdb5-1), &lorbdb5, &childinfo)
			goblas.Dscal(p, &negone, phantom, func() *int { y := 1; return &y }())
			Dlarfgp(p, phantom.GetPtr(0), phantom.Off(1), func() *int { y := 1; return &y }(), taup1.GetPtr(0))
			Dlarfgp(toPtr((*m)-(*p)), phantom.GetPtr((*p)+1-1), phantom.Off((*p)+2-1), func() *int { y := 1; return &y }(), taup2.GetPtr(0))
			theta.Set(i-1, math.Atan2(phantom.Get(0), phantom.Get((*p)+1-1)))
			c = math.Cos(theta.Get(i - 1))
			s = math.Sin(theta.Get(i - 1))
			phantom.Set(0, one)
			phantom.Set((*p)+1-1, one)
			Dlarf('L', p, q, phantom, func() *int { y := 1; return &y }(), taup1.GetPtr(0), x11, ldx11, work.Off(ilarf-1))
			Dlarf('L', toPtr((*m)-(*p)), q, phantom.Off((*p)+1-1), func() *int { y := 1; return &y }(), taup2.GetPtr(0), x21, ldx21, work.Off(ilarf-1))
		} else {
			Dorbdb5(toPtr((*p)-i+1), toPtr((*m)-(*p)-i+1), toPtr((*q)-i+1), x11.Vector(i-1, i-1-1), func() *int { y := 1; return &y }(), x21.Vector(i-1, i-1-1), func() *int { y := 1; return &y }(), x11.Off(i-1, i-1), ldx11, x21.Off(i-1, i-1), ldx21, work.Off(iorbdb5-1), &lorbdb5, &childinfo)
			goblas.Dscal(toPtr((*p)-i+1), &negone, x11.Vector(i-1, i-1-1), func() *int { y := 1; return &y }())
			Dlarfgp(toPtr((*p)-i+1), x11.GetPtr(i-1, i-1-1), x11.Vector(i+1-1, i-1-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1))
			Dlarfgp(toPtr((*m)-(*p)-i+1), x21.GetPtr(i-1, i-1-1), x21.Vector(i+1-1, i-1-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1))
			theta.Set(i-1, math.Atan2(x11.Get(i-1, i-1-1), x21.Get(i-1, i-1-1)))
			c = math.Cos(theta.Get(i - 1))
			s = math.Sin(theta.Get(i - 1))
			x11.Set(i-1, i-1-1, one)
			x21.Set(i-1, i-1-1, one)
			Dlarf('L', toPtr((*p)-i+1), toPtr((*q)-i+1), x11.Vector(i-1, i-1-1), func() *int { y := 1; return &y }(), taup1.GetPtr(i-1), x11.Off(i-1, i-1), ldx11, work.Off(ilarf-1))
			Dlarf('L', toPtr((*m)-(*p)-i+1), toPtr((*q)-i+1), x21.Vector(i-1, i-1-1), func() *int { y := 1; return &y }(), taup2.GetPtr(i-1), x21.Off(i-1, i-1), ldx21, work.Off(ilarf-1))
		}

		goblas.Drot(toPtr((*q)-i+1), x11.Vector(i-1, i-1), ldx11, x21.Vector(i-1, i-1), ldx21, &s, toPtrf64(-c))
		Dlarfgp(toPtr((*q)-i+1), x21.GetPtr(i-1, i-1), x21.Vector(i-1, i+1-1), ldx21, tauq1.GetPtr(i-1))
		c = x21.Get(i-1, i-1)
		x21.Set(i-1, i-1, one)
		Dlarf('R', toPtr((*p)-i), toPtr((*q)-i+1), x21.Vector(i-1, i-1), ldx21, tauq1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work.Off(ilarf-1))
		Dlarf('R', toPtr((*m)-(*p)-i), toPtr((*q)-i+1), x21.Vector(i-1, i-1), ldx21, tauq1.GetPtr(i-1), x21.Off(i+1-1, i-1), ldx21, work.Off(ilarf-1))
		if i < (*m)-(*q) {
			s = math.Sqrt(math.Pow(goblas.Dnrm2(toPtr((*p)-i), x11.Vector(i+1-1, i-1), func() *int { y := 1; return &y }()), 2) + math.Pow(goblas.Dnrm2(toPtr((*m)-(*p)-i), x21.Vector(i+1-1, i-1), func() *int { y := 1; return &y }()), 2))
			phi.Set(i-1, math.Atan2(s, c))
		}

	}

	//     Reduce the bottom-right portion of X11 to [ I 0 ]
	for i = (*m) - (*q) + 1; i <= (*p); i++ {
		Dlarfgp(toPtr((*q)-i+1), x11.GetPtr(i-1, i-1), x11.Vector(i-1, i+1-1), ldx11, tauq1.GetPtr(i-1))
		x11.Set(i-1, i-1, one)
		Dlarf('R', toPtr((*p)-i), toPtr((*q)-i+1), x11.Vector(i-1, i-1), ldx11, tauq1.GetPtr(i-1), x11.Off(i+1-1, i-1), ldx11, work.Off(ilarf-1))
		Dlarf('R', toPtr((*q)-(*p)), toPtr((*q)-i+1), x11.Vector(i-1, i-1), ldx11, tauq1.GetPtr(i-1), x21.Off((*m)-(*q)+1-1, i-1), ldx21, work.Off(ilarf-1))
	}

	//     Reduce the bottom-right portion of X21 to [ 0 I ]
	for i = (*p) + 1; i <= (*q); i++ {
		Dlarfgp(toPtr((*q)-i+1), x21.GetPtr((*m)-(*q)+i-(*p)-1, i-1), x21.Vector((*m)-(*q)+i-(*p)-1, i+1-1), ldx21, tauq1.GetPtr(i-1))
		x21.Set((*m)-(*q)+i-(*p)-1, i-1, one)
		Dlarf('R', toPtr((*q)-i), toPtr((*q)-i+1), x21.Vector((*m)-(*q)+i-(*p)-1, i-1), ldx21, tauq1.GetPtr(i-1), x21.Off((*m)-(*q)+i-(*p)+1-1, i-1), ldx21, work.Off(ilarf-1))
	}
}
